"""Guard test for the module-layout move series.

Each PR in the structural refactor moves one or more modules to a new
dotted path and leaves a lazy PEP 562 forwarder (``_shim.forwarder``) at
the old path, so external callers keep working with a deprecation warning.
Internally, though, nothing should still use the old path: a shim
*resolves* attribute access, so a stale
``monkeypatch.setattr("zotero_mcp.utils.is_local_mode", ...)`` would keep
succeeding even after ``is_local_mode`` moved away — it would just patch an
attribute nothing calls anymore, and the test would go on "passing" while
testing nothing. The same is true of a stray ``import zotero_mcp.utils``
left over from before a move.

This file scans the tree for exactly that: any quoted ``"zotero_mcp...."``
string or ``from``/``import`` statement whose dotted path is a shim's old
path (or a deeper attribute reached through it). It fails loudly, listing
every hit as ``file:line: old -> use new``, so a move PR can't leave a
stale reference behind. It does not scan itself -- see ``_test_files``.
"""

import re
from pathlib import Path

import pytest

# Old dotted path -> new dotted path. Each move PR appends its own row(s)
# here as part of moving a module; this map is the single source of truth
# both guard tests below check against. Empty in PR 0 (scaffolding) because
# nothing has moved yet: with no rows, the two whole-tree guards below scan
# nothing at all and pass vacuously. What keeps the scan itself honest until
# PR 1 adds a row is `test_whole_tree_scan_finds_and_formats_real_hits`,
# which runs it over the real tree against a synthetic row.
SHIMS: dict[str, str] = {}

# This file lives at the top level of tests/ (never inside a subpackage —
# see the module docstring in tests/conftest.py for why tests/ has no
# __init__.py), so parents[1] is the repo root. If this file is ever moved
# into a subdirectory of tests/, this needs to become parents[2].
_GUARD_FILE = Path(__file__).resolve()
_REPO_ROOT = _GUARD_FILE.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"

_QUOTED_PATH_RE = re.compile(r"""(['"])(zotero_mcp(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\1""")
_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+(.+)$")
_IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")


def _shim_match(path: str, old: str, src_root: Path | None = None) -> bool:
    """Return True if a reference to `path` resolves through the shim at `old`.

    A bare reference to `old` itself always matches. A deeper reference
    ``old.<name>`` matches too, *unless* `<name>` is a real submodule of the
    package at `old` in the source tree rooted at `src_root` (the ``src/``
    directory containing the ``zotero_mcp`` package) — i.e.
    ``<old-as-dir>/<name>.py`` or ``<old-as-dir>/<name>/__init__.py`` exists
    on disk. That case is a legitimate reference to a sibling module under
    the now-real package and must not be flagged.

    This is decided from the filesystem, not from a shim's `new` value,
    because a package-as-shim entry like ``zotero_mcp.cli`` ->
    ``zotero_mcp.cli.manage`` gets *other* real submodules too
    (``standalone``, ``envelope``, ...) that are not `new` but must not be
    flagged either — only `new`'s own submodule would be an under-exemption
    that flags every other legitimate submodule reference.

    A plain-module shim (`old` isn't a directory in the source tree at
    all, e.g. it was fully replaced by an unrelated new path) has no real
    submodules, so every ``old.<name>`` reference is flagged — which is
    also what happens when `src_root` is omitted, so callers that don't
    care about submodule probing (e.g. plain-module unit tests) don't need
    to build a fake source tree.
    """
    if path == old:
        return True
    if not path.startswith(old + "."):
        return False
    if src_root is not None:
        name = path[len(old) + 1 :].split(".", 1)[0]
        old_dir = src_root.joinpath(*old.split("."))
        if (old_dir / f"{name}.py").is_file() or (old_dir / name / "__init__.py").is_file():
            return False
    return True


def _candidates_from_line(line: str) -> list[str]:
    """Return the `zotero_mcp....` dotted paths a single line of source
    text references, either as a quoted string literal or as an import
    target (``import a.b.c[, ...]`` / ``from a.b import c[, ...]``).

    For ``from X import Y``, the candidate is the combined ``X.Y`` (what is
    actually being reached), not bare ``X`` — otherwise a perfectly good
    ``from zotero_mcp.cli import manage`` would be indistinguishable from
    the deprecated ``import zotero_mcp.cli`` itself. A wildcard import
    (``from X import *``) can't be resolved that way, so it falls back to
    the bare module path.
    """
    candidates: list[str] = []

    for _quote, path in _QUOTED_PATH_RE.findall(line):
        candidates.append(path)

    from_match = _FROM_IMPORT_RE.match(line)
    if from_match:
        module, names_part = from_match.group(1), from_match.group(2)
        if module.startswith("zotero_mcp"):
            names_part = names_part.split("#", 1)[0].strip().strip("()")
            for piece in names_part.split(","):
                name = piece.strip().split(" as ")[0].strip()
                if not name:
                    continue
                candidates.append(module if name == "*" else f"{module}.{name}")
        return candidates

    import_match = _IMPORT_RE.match(line)
    if import_match:
        imports_part = import_match.group(1).split("#", 1)[0]
        for piece in imports_part.split(","):
            dotted = piece.strip().split(" as ")[0].strip()
            if dotted.startswith("zotero_mcp"):
                candidates.append(dotted)

    return candidates


def _violations_in_line(line: str, shims: dict[str, str], src_root: Path | None = None) -> list[tuple[str, str]]:
    """Return the `(old, new)` shim rows that a line of source text references.

    `src_root` (the ``src/`` directory) is forwarded to `_shim_match` so it
    can tell a shim's old path apart from a real sibling submodule; omit it
    for cases that don't turn on that distinction.
    """
    if not shims:
        return []
    hits: list[tuple[str, str]] = []
    for candidate in _candidates_from_line(line):
        for old, new in shims.items():
            if _shim_match(candidate, old, src_root):
                hits.append((old, new))
    return hits


def _module_dotted_path(file: Path, src_root: Path) -> str:
    """Return the dotted module path that `file` (under `src_root`, the
    ``src/`` directory) implements. ``__init__.py`` implements its
    containing package, not a submodule literally named ``__init__``."""
    parts = list(file.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _find_violations(files: list[Path], shims: dict[str, str], repo_root: Path, src_root: Path) -> list[str]:
    """Scan `files` for references to a `shims` old-path and return one
    formatted ``file:line: old -> use new`` message per hit."""
    if not shims:
        return []
    messages: list[str] = []
    for file in files:
        rel = file.relative_to(repo_root)
        text = file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for old, new in _violations_in_line(line, shims, src_root):
                messages.append(f"{rel}:{lineno}: {old} -> use {new}")
    return messages


def _test_files() -> list[Path]:
    """Every file under `tests/` the guard scans — that is, all of them
    except this one.

    This file is exempt because it *is* the map: `SHIMS`, the unit tests
    below and the docstrings all quote old paths deliberately, and the
    scanner cannot tell those apart from a stale reference. Without the
    exemption, the PR that adds the first row could not add it — a
    ``zotero_mcp.client`` row matches 7 lines of this file on its own, a
    ``zotero_mcp.cli`` row 16. Nothing else is exempt: a real reference in
    any other test file must still fail the build.
    """
    return [f for f in sorted((_REPO_ROOT / "tests").rglob("*.py")) if f.resolve() != _GUARD_FILE]


def _source_files(shims: dict[str, str]) -> list[Path]:
    """Every production module the guard scans.

    `_shim.py` and the shim stub file for each row (the file implementing
    the old dotted path itself, which legitimately mentions both the old
    and new paths to build the forwarder) are excluded.
    """
    return [
        f
        for f in sorted((_SRC_ROOT / "zotero_mcp").rglob("*.py"))
        if f.name != "_shim.py" and _module_dotted_path(f, _SRC_ROOT) not in shims
    ]


def test_no_test_patches_or_imports_a_shim_path():
    """No test file may patch or import a shim's old dotted path.

    Doing so would look like it still exercises the moved code — a
    ``monkeypatch.setattr("<old>.<name>", ...)`` still succeeds, it just
    patches an attribute nothing calls anymore — so this must fail the
    build instead of silently testing nothing.
    """
    violations = _find_violations(_test_files(), SHIMS, _REPO_ROOT, _SRC_ROOT)
    assert not violations, "tests reference a shim's old path; update to the new path:\n" + "\n".join(violations)


def test_no_source_module_imports_a_shim_path():
    """No production module may import or reference a shim's old dotted
    path either — internal callers must use the new location directly,
    since the shim only exists to keep *external* callers working.
    """
    violations = _find_violations(_source_files(SHIMS), SHIMS, _REPO_ROOT, _SRC_ROOT)
    assert not violations, "source references a shim's old path; update to the new path:\n" + "\n".join(violations)


# A row that is not in `SHIMS` and whose old path is still all over the real
# tree — `zotero_mcp.utils` is PR 9's move, the last one in the series. It
# stands in for a real row so the whole-tree scan can be exercised while
# `SHIMS` is empty.
_SYNTHETIC_ROW = {"zotero_mcp.utils": "zotero_mcp.formatting.display"}
_VIOLATION_RE = re.compile(
    r"^(?P<file>[^:]+\.py):(?P<line>\d+): zotero_mcp\.utils -> use zotero_mcp\.formatting\.display$"
)


def test_whole_tree_scan_finds_and_formats_real_hits():
    """The whole-tree scan must actually read the tree and report hits.

    With `SHIMS` empty, `_find_violations` returns immediately and the two
    guards above read **no files at all** — they pass without touching a
    single line of the tree, and would go on passing if the file walk, the
    line scanner or the message format broke. This runs the same scan over
    the same real files against `_SYNTHETIC_ROW`, so the machinery is
    proven before PR 1 adds the first real row.

    The count is deliberately not asserted: it drifts every time a module
    moves. What is asserted is that hits exist in both trees, that every
    message has the ``file:line: old -> use new`` shape a move-PR author
    has to act on, and that the file and line each one names really do
    hold a reference to the old path.
    """
    files = _test_files() + _source_files(_SYNTHETIC_ROW)
    messages = _find_violations(files, _SYNTHETIC_ROW, _REPO_ROOT, _SRC_ROOT)

    assert messages, (
        f"scanning {len(files)} real files for {list(_SYNTHETIC_ROW)[0]} found nothing; "
        "the whole-tree scan is not reading the tree"
    )

    malformed = [m for m in messages if not _VIOLATION_RE.match(m)]
    assert not malformed, "violations must read 'file:line: old -> use new':\n" + "\n".join(malformed)

    assert any(m.startswith("src/") for m in messages), "no source-tree hits"
    assert any(m.startswith("tests/") for m in messages), "no test-tree hits"

    for message in messages:
        found = _VIOLATION_RE.match(message)
        line = (_REPO_ROOT / found["file"]).read_text(encoding="utf-8").splitlines()[int(found["line"]) - 1]
        # Checked in two halves, not as the literal dotted path: a legitimate
        # hit can spell it across the line (`from zotero_mcp import utils`).
        assert "zotero_mcp" in line and "utils" in line, (
            f"{message} points at a line that does not reference the old path: {line!r}"
        )


def test_the_guard_file_is_the_only_exempt_test_file():
    """The self-exemption must cover this file and nothing else.

    `SHIMS`, the fixtures below and the docstrings quote old paths on
    purpose, so this file has to be out of scope — but a blanket exemption
    (a whole directory, a filename pattern) would quietly stop scanning
    real test files, which is the failure this whole module exists to
    prevent.
    """
    scanned = _test_files()
    all_test_files = sorted((_REPO_ROOT / "tests").rglob("*.py"))

    assert [f for f in all_test_files if f not in scanned] == [_GUARD_FILE], (
        "exactly one test file may be exempt from the scan"
    )
    assert _find_violations([_GUARD_FILE], {"zotero_mcp.client": "zotero_mcp.backends.api"}, _REPO_ROOT, _SRC_ROOT), (
        "this file no longer quotes any old path, so the exemption is load-bearing for nothing -- "
        "drop it and scan this file like the rest"
    )


class TestShimMatchPlainModule:
    """A plain module shim (old path fully replaced by an unrelated new
    path, e.g. `zotero_mcp.client` -> `zotero_mcp.backends.api`) must flag
    any reference to the old path or an attribute reached through it, and
    must never flag the new path.
    """

    SHIMS = {"zotero_mcp.client": "zotero_mcp.backends.api"}

    def test_bare_import_of_old_module_is_flagged(self):
        assert _violations_in_line("import zotero_mcp.client", self.SHIMS) == [
            ("zotero_mcp.client", "zotero_mcp.backends.api")
        ]

    def test_from_import_of_old_module_attribute_is_flagged(self):
        assert _violations_in_line("from zotero_mcp.client import get_zotero_client", self.SHIMS) == [
            ("zotero_mcp.client", "zotero_mcp.backends.api")
        ]

    def test_quoted_patch_string_for_old_path_is_flagged(self):
        line = 'monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: None)'
        assert _violations_in_line(line, self.SHIMS) == [("zotero_mcp.client", "zotero_mcp.backends.api")]

    def test_wildcard_import_of_old_module_is_flagged(self):
        assert _violations_in_line("from zotero_mcp.client import *", self.SHIMS) == [
            ("zotero_mcp.client", "zotero_mcp.backends.api")
        ]

    def test_new_path_is_never_flagged(self):
        assert _violations_in_line("from zotero_mcp.backends.api import get_zotero_client", self.SHIMS) == []
        assert (
            _violations_in_line('monkeypatch.setattr("zotero_mcp.backends.api.get_zotero_client", x)', self.SHIMS) == []
        )

    def test_unrelated_module_is_not_flagged(self):
        assert _violations_in_line("import zotero_mcp.utils", self.SHIMS) == []

    def test_is_local_mode_example_from_the_brief_is_flagged(self):
        """The brief's own canonical example row. Unaffected by the
        filesystem-probe fix in `_shim_match`, since `zotero_mcp.utils` has
        no real submodule named `is_local_mode` — a plain-module shim's
        `old.<name>` references stay flagged regardless."""
        shims = {"zotero_mcp.utils": "zotero_mcp.formatting.display"}
        assert _violations_in_line("from zotero_mcp.utils import is_local_mode", shims) == [
            ("zotero_mcp.utils", "zotero_mcp.formatting.display")
        ]


@pytest.fixture
def cli_src_root(tmp_path):
    """A fake `src/` tree shaped like PR 4 leaves `zotero_mcp.cli`: a real
    package whose `__init__.py` shims the old flat-module attributes, next
    to real sibling submodules (`standalone`, `envelope`) that are not the
    shim's `new` value but must never be flagged either."""
    cli_dir = tmp_path / "src" / "zotero_mcp" / "cli"
    cli_dir.mkdir(parents=True)
    for name in ("__init__", "manage", "standalone", "envelope"):
        (cli_dir / f"{name}.py").write_text("")
    return tmp_path / "src"


class TestShimMatchPackageAsShim:
    """A package-as-shim entry (old path becomes a real package whose
    __init__.py forwards to a submodule one level under it, e.g.
    `zotero_mcp.cli` -> `zotero_mcp.cli.manage`) must flag the bare old
    path and any attribute of it *except* a real submodule — any real
    submodule, not just the one `new` names — so that genuine references
    to every submodule under the same package are never flagged.
    """

    SHIMS = {"zotero_mcp.cli": "zotero_mcp.cli.manage"}

    def test_bare_import_of_old_package_is_flagged(self):
        assert _violations_in_line("import zotero_mcp.cli", self.SHIMS) == [("zotero_mcp.cli", "zotero_mcp.cli.manage")]

    def test_moved_attribute_reached_through_old_package_is_flagged(self):
        assert _violations_in_line("from zotero_mcp.cli import main", self.SHIMS) == [
            ("zotero_mcp.cli", "zotero_mcp.cli.manage")
        ]

    def test_quoted_patch_string_through_old_package_is_flagged(self):
        line = 'monkeypatch.setattr("zotero_mcp.cli.setup_zotero_environment", lambda: None)'
        assert _violations_in_line(line, self.SHIMS) == [("zotero_mcp.cli", "zotero_mcp.cli.manage")]

    def test_real_submodule_import_is_not_flagged(self, cli_src_root):
        assert _violations_in_line("from zotero_mcp.cli import manage", self.SHIMS, cli_src_root) == []
        assert _violations_in_line("import zotero_mcp.cli.manage", self.SHIMS, cli_src_root) == []
        assert _violations_in_line("import zotero_mcp.cli.manage as cli_manage", self.SHIMS, cli_src_root) == []

    def test_attribute_of_real_submodule_is_not_flagged(self, cli_src_root):
        assert _violations_in_line("from zotero_mcp.cli.manage import main", self.SHIMS, cli_src_root) == []
        assert (
            _violations_in_line('monkeypatch.setattr("zotero_mcp.cli.manage.main", x)', self.SHIMS, cli_src_root) == []
        )

    def test_sibling_submodules_are_not_flagged(self, cli_src_root):
        """Real siblings of the shim's `new` submodule (`standalone`,
        `envelope`, the shape PR 4 actually leaves behind in `cli/`) must
        not be flagged either — this is exactly the false positive the
        Critical finding caught: exempting only `new`'s literal submodule
        flagged every other legitimate submodule reference."""
        assert _violations_in_line("from zotero_mcp.cli import standalone", self.SHIMS, cli_src_root) == []
        assert _violations_in_line("from zotero_mcp.cli import envelope", self.SHIMS, cli_src_root) == []
        assert _violations_in_line("import zotero_mcp.cli.standalone", self.SHIMS, cli_src_root) == []
        assert _violations_in_line("import zotero_mcp.cli.envelope", self.SHIMS, cli_src_root) == []

    def test_bare_and_unknown_attribute_still_flagged_with_real_tree(self, cli_src_root):
        """The filesystem probe must not swallow genuine violations: the
        bare old package and any attribute that isn't a real submodule
        (a moved function/class) still get flagged even once `cli/` is a
        real package with real submodules on disk."""
        assert _violations_in_line("import zotero_mcp.cli", self.SHIMS, cli_src_root) == [
            ("zotero_mcp.cli", "zotero_mcp.cli.manage")
        ]
        assert _violations_in_line("from zotero_mcp.cli import some_function", self.SHIMS, cli_src_root) == [
            ("zotero_mcp.cli", "zotero_mcp.cli.manage")
        ]


@pytest.fixture
def semantic_search_src_root(tmp_path):
    """A fake `src/` tree for `zotero_mcp.semantic_search`, shaped like PR 3:
    the shim's `new` submodule (`engine`) plus a sibling that is itself a
    subpackage (`batch/`) rather than a plain `.py` file — covering the
    `<name>/__init__.py` branch of the submodule probe."""
    ss_dir = tmp_path / "src" / "zotero_mcp" / "semantic_search"
    ss_dir.mkdir(parents=True)
    (ss_dir / "__init__.py").write_text("")
    (ss_dir / "engine.py").write_text("")
    batch_dir = ss_dir / "batch"
    batch_dir.mkdir()
    (batch_dir / "__init__.py").write_text("")
    return tmp_path / "src"


class TestShimMatchSubpackageSubmodule:
    """A real submodule of a shimmed package can itself be a subpackage (a
    directory with `__init__.py`), not just a plain `.py` file — the probe
    must recognize both shapes, and must still flag a moved function name
    that happens to share the package with real submodules.
    """

    SHIMS = {"zotero_mcp.semantic_search": "zotero_mcp.semantic_search.engine"}

    def test_subpackage_submodule_is_not_flagged(self, semantic_search_src_root):
        assert (
            _violations_in_line("from zotero_mcp.semantic_search import batch", self.SHIMS, semantic_search_src_root)
            == []
        )
        assert (
            _violations_in_line("import zotero_mcp.semantic_search.batch", self.SHIMS, semantic_search_src_root) == []
        )

    def test_moved_function_attribute_is_still_flagged(self, semantic_search_src_root):
        """`get_zotero_client` isn't a submodule (no such file/dir exists
        under the fake `semantic_search/`), so it must still be flagged
        even though the package has real submodules on disk."""
        assert _violations_in_line(
            "from zotero_mcp.semantic_search import get_zotero_client", self.SHIMS, semantic_search_src_root
        ) == [("zotero_mcp.semantic_search", "zotero_mcp.semantic_search.engine")]


class TestModuleDottedPath:
    """`_module_dotted_path` must map a shim stub file back to the exact
    dotted key it would occupy in `SHIMS`, for both a plain module and a
    package's `__init__.py`.
    """

    def test_plain_module_file(self, tmp_path):
        src_root = tmp_path / "src"
        f = src_root / "zotero_mcp" / "client.py"
        assert _module_dotted_path(f, src_root) == "zotero_mcp.client"

    def test_package_init_file(self, tmp_path):
        src_root = tmp_path / "src"
        f = src_root / "zotero_mcp" / "cli" / "__init__.py"
        assert _module_dotted_path(f, src_root) == "zotero_mcp.cli"

    def test_nested_submodule_file(self, tmp_path):
        src_root = tmp_path / "src"
        f = src_root / "zotero_mcp" / "cli" / "manage.py"
        assert _module_dotted_path(f, src_root) == "zotero_mcp.cli.manage"
