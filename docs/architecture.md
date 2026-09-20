# Architecture

Where code goes in this package, and why. Read this before adding a module, adding a test file, or moving
anything — the rules below exist because breaking one of them has cost this project real time.

## Status of this document

> **The package tree is still flat.** `src/zotero_mcp/` is a directory of modules with two subpackages
> (`tools/` and `embeddings/`). The packages described below — `backends/`, `attachments/`,
> `metadata_sources/`, `semantic_search/`, `cli/`, `formatting/` — **do not exist yet**. They arrive one at a
> time, and the [shim table](#5-shims) in this document is empty because nothing has moved.

This is a target document, not a description of the current directory listing. Every table that follows
carries a **Status** column with one of two values:

| Value | Meaning |
|---|---|
| **now** | Exists in `src/zotero_mcp/` today. The rule is in force. |
| **target** | Does not exist yet. The rule applies to the PR that creates it. |

What is in the tree today:

```
src/zotero_mcp/
├── __init__.py  _version.py  _shim.py  _app.py  _context.py  server.py
├── toolsets.py  prompts.py  resources.py  config.py  config_light.py
├── schema.py  identifiers.py  search_semantics.py  utils.py  library.py
├── client.py  local_db.py  better_bibtex_client.py  webdav.py  scite_client.py
├── extract.py  fulltext_cache.py  pdf_utils.py  pdf_layout.py  epub_utils.py
├── pdfannots_helper.py  pdfannots_downloader.py  citation_import.py  html_metadata.py
├── semantic_search.py  chroma_client.py  batch_common.py  openai_batch.py  gemini_batch.py
├── cli.py  cli_standalone.py  cli_json.py  setup_helper.py  updater.py  skill_install.py
├── embeddings/        # provider adapters
├── tools/             # the MCP tool surface, with tools/_helpers.py and tools/write.py
├── data/  skills/     # package data, not Python packages
```

## 1. Package map

Each package has an **import-cost rule**, which is a statement about what its `__init__.py` is allowed to
pull in. The rule matters because `import zotero_mcp.<anything>` used to cost about 1.45 s and ~1480 modules
— see [§6](#6-the-import-cost-budget) — and a subpackage `__init__` that imports its own contents eagerly
puts that cost straight back.

| Package | What lives there | Import-cost rule | Status |
|---|---|---|---|
| package root (flat) | MCP wiring (`_app`, `_context`, `server`, `toolsets`, `prompts`, `resources`), vocabulary and config (`config`, `config_light`, `schema`, `identifiers`, `search_semantics`), `_shim`, `_version`, and — for now — `utils.py` and every module not yet claimed by a package below | `__init__.py` exports `__version__` eagerly and exposes `mcp` lazily (PEP 562); nothing else. `_version.py` never moves: `[tool.hatch.version]` reads it by path. | now |
| package root, after `utils.py` is split | `utils.py` is not replaced by a `utils/` package. Its process-level odds and ends stay flat as `distribution.py`, `paths.py`, `_stdio.py` and `search_variants.py`; its formatting functions go to `formatting/` and its backend selection to `backends/`. | Each stays stdlib-only, as `utils.py` largely is today. | target |
| `backends/` | Systems that hold library data: `library.py` (the `Protocol` and its fallback), `api.py`, `sqlite.py`, `bibtex.py`, `webdav.py`, `scite.py`, plus backend `selection.py` and `pagination.py` | Empty `__init__.py`. No pyzotero, no sqlite connection, no network client at import time. `pagination.py` must stay stdlib-only: `cli_standalone.py` imports it at top level and defers pyzotero. | target |
| `attachments/` | Zotero's word for the files under an item: their text (`extract.py`, `fulltext_cache.py`), PDF and EPUB access (`pdf.py`, `pdf_layout.py`, `epub.py`), annotation import (`pdfannots.py`, `pdfannots_installer.py`), and where to find an open-access copy (`openaccess.py`) | Empty `__init__.py`. PyMuPDF (`fitz`) is an optional dependency and must be imported inside functions, never at module scope in the `__init__`. | target |
| `metadata_sources/` | Where bibliographic metadata comes from outside Zotero: `crossref.py`, `arxiv.py`, `isbn.py`, `webpage.py`, `citation_import.py`, `html_metadata.py` | Empty `__init__.py`. No `requests` at import time. | target |
| `semantic_search/` | The feature's own name in the CLI and the config file: `engine.py`, `chroma.py`, `batch/`, `embeddings/`, and the pieces split out of the engine (`lock`, `chunking`, `reranker`, `settings`, `documents`, `sync_state`, `sources/`, `indexer`, `query`, …) | `__init__.py` is a lazy forwarder to `engine`; it imports nothing. This is the strictest rule in the tree: `_app.py` and `cli.py` both decide from config *whether* to use semantic search, and both gates are only meaningful while ChromaDB is still unimported (#485). | target |
| `cli/` | Console commands: `manage.py`, `standalone.py`, `envelope.py`, `wizard.py`, `updater.py`, `skill_install.py`, `semantic_db.py`, and `__main__.py` | `__init__.py` forwards `main` permanently (it is an entry-point target, not a deprecation shim) and imports nothing else. The `#485` config gates live here. | target |
| `formatting/` | Turning field content into display strings: `names.py`, `markup.py`, `display.py` | Empty `__init__.py`, and the modules themselves stay stdlib-only. | target |
| `tools/` | The MCP tool surface, one module per tool group. Today that includes the single modules `_helpers.py` and `write.py`; both become packages (`_helpers/`, `write/`) at the same dotted paths. | **The one heavy package, deliberately.** Importing it registers every tool by side effect and pulls in FastMCP, pydantic and pyzotero. Nothing outside `server.py` and the CLI's `_import_tools()` may import it at module scope. | now |
| `embeddings/` | Embedding-provider adapters. Moves under `semantic_search/` unchanged. | Inherits `semantic_search/`'s rule once it moves. | now |
| `data/`, `skills/` | Package data — a fields table and the `zotero-cli` skill. Not Python packages, and they stay at the package root because docs, scripts and tests address them there (`schema.py` and `skill_install.py` both resolve them relative to `__file__`). | n/a | now |

A reader should be able to guess a package's contents from its name without opening it. `documents`, `text`,
`semantic` and `ingest` were considered and rejected for failing that test.

## 2. Naming rules

**Packages** use Zotero's own vocabulary where one exists (`attachments`, `collections`, `annotations`) and
otherwise name their contents (`backends`, `metadata_sources`, `formatting`). A feature keeps the name it has
in the CLI and the config file (`semantic_search`).

**Modules** are a noun naming what the module *is*, never how it is used:

- A service-named module is that service's client — `backends/api.py`, `backends/sqlite.py`,
  `backends/bibtex.py`, `semantic_search/chroma.py`, `metadata_sources/crossref.py`.
- A format-named module is that format's parser or writer — `attachments/pdf.py`, `attachments/epub.py`,
  `formatting/markup.py`.

**Banned suffixes: `_utils`, `_helpers`, `_common`, `_misc`.** A module that needs one has two concerns; split
it. Three exceptions are tolerated, each for a stated reason:

| Exception | Why it is allowed |
|---|---|
| `tools/_helpers/` | Its import path is fixed and does not change when the module becomes a package: 278 quoted patch strings across 27 test files target `zotero_mcp.tools._helpers.*`. Its `__init__` re-exports every name, so the late-binding `_helpers.X(...)` call style keeps resolving. |
| `semantic_search/batch/common.py` | Not a grab bag — it is the abstract base class the two provider modules subclass. |
| `tools/write/_common.py` | Private to its package (leading underscore) and not importable as a seam from outside it. |

**A leading underscore means internal to this package** and may change without a shim: `_app`, `_context`,
`_shim`, `_stdio`, `_env`, `tools/_helpers/`, `write/_common`.

**Never name a submodule after a function its package `__init__` re-exports.** A PEP 562 `__getattr__`
returning a function named `main` and a submodule named `main` collide: after `import zotero_mcp.cli.main`
the import system binds the *module* to that attribute, and the entry point silently gets the wrong object.
This is why the `cli` command module is `manage.py`, not `main.py`.

## 3. How tools reach helpers, and why `server.py` is a facade

`server.py` contains no code of its own — zero `def` and zero `class` statements. It does two things:

1. `import zotero_mcp.tools`, whose only job is to register every `@mcp.tool` by import side effect.
2. Re-export ~108 names from `_app`, `client`, `tools._helpers` and elsewhere, so that
   `from zotero_mcp.server import X` keeps working. 40 test files do `from zotero_mcp import server` and
   reach helpers through it.

Keeping it a facade is what makes the packages below it movable: a module can change its dotted path without
any of those 40 test files noticing, as long as `server.py`'s re-export list is updated in the same commit.
Do not add logic to `server.py`; add it to the package that owns the behaviour and re-export it here if a
caller needs it.

**Tools reach helpers by attribute lookup on the module, not by importing the function.** Every tool module
does `from zotero_mcp.tools import _helpers` and then calls `_helpers._get_write_client(ctx)`. The obvious
alternative, `from zotero_mcp.tools._helpers import _get_write_client`, binds the function object at import
time, and a test patching `zotero_mcp.tools._helpers._get_write_client` would then never be seen. This is the
**one seam rule**:

> A name that tests patch on module M must be looked up on M at call time by every internal caller.

When a split puts a patched callee and its caller in different files, the caller resolves the callee through
the package rather than importing it:

```python
def _pkg():
    from zotero_mcp.tools import _helpers
    return _helpers

...
_pkg()._maybe_upload_to_webdav(path)
```

A shim at an old path does **not** rescue a string patch. `monkeypatch.setattr` sets the attribute on
whichever module object the path names, and a caller that looks the name up somewhere else never sees it —
the patch succeeds, the test passes, and nothing was tested. The patch seams that must keep resolving are
`zotero_mcp.tools._helpers._get_write_client` (~207 occurrences), `zotero_mcp.client.get_zotero_client`
(~116), `zotero_mcp.tools.write.requests.get`, `zotero_mcp.tools._helpers._try_attach_oa_pdf`,
`zotero_mcp.utils.is_local_mode`, `zotero_mcp.tools._helpers.find_existing_items`,
`zotero_mcp.cli_standalone.setup_zotero_environment` and `._import_tools`, and
`zotero_mcp.tools.write._time.sleep`.

## 4. Test layout

`tests/` mirrors `src/zotero_mcp/`. A test file goes where the code whose behaviour it asserts lives; if it
calls two layers, the one whose functions it calls *directly* wins.

| Kind of test | Where it goes |
|---|---|
| Module test | `tests/<package>/test_<module>.py`, renamed with the module (`test_local_db.py` becomes `tests/backends/test_sqlite.py`) |
| Behaviour test | `tests/<package>/test_<module>_<behaviour>.py`. It must **not** start with `test_<package>_` — that prefix duplicates the directory. |
| MCP tool test | `tests/tools/test_<tool_name>.py`, and `tests/tools/write/` for the write tools |
| Issue regression | `test_issue_<n>_<slug>.py`, next to the module it pins; at the root if it crosses layers |
| Cross-cutting | Stays at the root: `test_lightweight_imports`, `test_context_cost_claim`, `test_description_tokens`, `test_skill_reference_current`, `test_toolsets`, `test_server_*`, `test_lifespan`, `test_module_layout` |
| Live / fixtures | `tests/live/`, `tests/fixtures/` and `tests/_search_corpus.py` stay where they are |

**`tests/` has no `__init__.py` and must never get one.** 68 test files do `from conftest import ...`, which
works only because pytest puts a rootdir-level test directory on `sys.path` as a plain script directory. Add
`tests/__init__.py` and `conftest` stops being importable by that name, all at once.

**Every `tests/<package>/` subdirectory does get an empty `__init__.py`**, so that duplicate basenames
(`tests/backends/test_pagination.py` and `tests/tools/test_pagination.py`) are legal under pytest's default
import mode. The precedent is `tests/live/__init__.py`, which is a zero-byte file and exists for exactly this
reason.

Two consequences follow from those subpackage `__init__.py` files. Neither fails loudly, so check both when
adding a directory:

- A directory named `tests/cli/` with an `__init__.py` makes `cli` importable as a *top-level* package while
  `tests/` is on `sys.path`. Before adding one, check that the name is not already resolvable:
  `python -c "import importlib.util as u; print([n for n in ('backends','attachments','metadata_sources','semantic_search','cli','formatting','tools') if u.find_spec(n)])"`
  must print `[]`.
- A test that moves one level deeper must not compute the repo root with `parents[1]`. `tests/conftest.py`
  can, because it sits at the top of `tests/`; a file in `tests/backends/` needs `parents[2]`.

**Fixtures are reached through `conftest.FIXTURES_DIR`**, never `Path(__file__).parent / "fixtures"` — tests
live at several depths and only the constant stays correct when one of them moves. A `fixtures_dir` fixture
returns the same path for tests that prefer to take it as an argument.

**`tests/conftest.py` isolates `HOME` at import time**, before any fixture runs, because `config.py` and
`client.py` compute `~/.config/zotero-mcp/...` at *their* import time. It points `HOME` and `USERPROFILE` at a
fresh temporary directory and removes it at exit. Without this, `update_database()` flocks the developer's
real `~/.config/zotero-mcp/update.lock` — which a running `zotero-mcp` server holds — and 51 tests fail for a
reason that has nothing to do with the change under test. Live tests need the real home, so the isolation is
skipped when `ZOTERO_MCP_LIVE_TESTS=1`:

```bash
python -m pytest --ignore=tests/test_lifespan.py -q       # isolated HOME, the normal case
ZOTERO_MCP_LIVE_TESTS=1 python -m pytest tests/live -q    # real HOME, real credentials
```

The same conftest also inserts this checkout's `src/` at the front of `sys.path` and purges `zotero_mcp*`
from `sys.modules`, so the suite exercises the tree it lives in rather than whatever an editable install
resolves to. That purge is also why the shim gate is armed from `conftest` and not from a `-W` flag —
see [§5](#the-movedmodulewarning-gate).

## 5. Shims

A module that moves leaves a **lazy PEP 562 forwarder** at its old path, built by `zotero_mcp._shim.forwarder`:

```python
# src/zotero_mcp/client.py, after the move
from zotero_mcp._shim import forwarder

__getattr__ = forwarder("zotero_mcp.client", "zotero_mcp.backends.api")
```

`forwarder(old, new, removed_in="0.14.0")` returns a module-level `__getattr__`. Four properties matter:

- **Importing the old path costs nothing.** Nothing is imported until a name is actually used, so a stale
  `import zotero_mcp.client` does not drag the new module — or its dependencies — into a cold interpreter.
- **Each *use* warns**, at the caller's line, with `MovedModuleWarning` (a `DeprecationWarning` subclass).
  Because `pyproject.toml` sets `filterwarnings = ["ignore::DeprecationWarning"]`, the warning is invisible in
  a normal run; in the test suite it is an **error**, armed by `tests/conftest.py` (see
  [the gate](#the-movedmodulewarning-gate) below).
- **Patching a name on a shim does not reach the new module.** `monkeypatch.setattr("zotero_mcp.client.get_zotero_client", ...)`
  sets an attribute on the shim module object; callers inside `backends/api.py` look the name up on their own
  module and never see it. That is the whole reason `tests/test_module_layout.py` exists.
- **`from <old path> import *` forwards nothing, silently.** A star-import (and `dir()`) reads the module's
  own `__dict__` and its `__all__`, neither of which a `__getattr__` contributes to — and the dunder guard
  makes `__all__` itself unreachable — so the caller binds only the shim file's own globals (`forwarder`),
  gets none of the moved names and no `MovedModuleWarning`, then fails later with a `NameError`. Nothing
  in-tree star-imports, but an external caller that does gets no deprecation notice at all: worth a release
  note when a widely imported module moves.

`new` may also be a `{name: module_path}` map, for a module that was split across several new homes.

`tests/test_module_layout.py` holds a `SHIMS` dict — old dotted path to new dotted path — and uses it for two
guards: no test file may patch or import a shim's old path, and no production module may either. The old
paths are for *external* callers only. Add a module's row to `SHIMS` in the same commit that adds its
forwarder; the guards then fail with `file:line: old -> use new` for every internal reference left behind.

**`tests/test_module_layout.py` does not scan itself; every other file under `tests/` is in scope.** The map,
the unit-test fixtures below it and the docstrings all quote old paths deliberately, and the scanner cannot
tell those from a stale reference — a `zotero_mcp.client` row matches 7 lines of that file, a `zotero_mcp.cli`
row 16, so without the exemption no PR could add its own row. The exemption is exactly one file:
`test_the_guard_file_is_the_only_exempt_test_file` asserts that, and that the file would be flagged if it
were scanned. Keep a real reference out of it — if you need to demonstrate one, write it in a unit-test
fixture with a row that is not in `SHIMS`, as the tests there do.

**With `SHIMS` empty the two whole-tree guards read no files at all** and pass vacuously, so until PR 1 adds
a row the scan itself is proven by `test_whole_tree_scan_finds_and_formats_real_hits`, which runs it over the
real tree against a synthetic `zotero_mcp.utils` row and checks the hits and their formatting.

**A row whose old path becomes a real package matches by prefix, minus the real submodules.** When a flat
module becomes a package of the same name — the `cli` case, where `zotero_mcp.cli` -> `zotero_mcp.cli.manage`
— the old path keeps getting new, legitimate submodules under it. The guards resolve this from the
filesystem rather than from the row: any name that exists as `<old-as-dir>/<name>.py` or
`<old-as-dir>/<name>/__init__.py` in `src/` is exempt, not just the one `new` points at. So under that row:

| Reference | Flagged? | Why |
|---|---|---|
| `import zotero_mcp.cli` | yes | the bare old path |
| `from zotero_mcp.cli import main` | yes | `main` is a moved function, not a file on disk |
| `monkeypatch.setattr("zotero_mcp.cli.setup_zotero_environment", ...)` | yes | same, reached as a quoted string |
| `from zotero_mcp.cli import standalone` | no | `cli/standalone.py` exists — a real sibling, and not the row's `new` |
| `from zotero_mcp.cli.manage import main` | no | the new path |

Exempting only the row's own `new` submodule would flag every other real submodule in the package, so the
probe deliberately exempts all of them. The cost is that a moved *function* whose name happens to collide
with a real submodule would slip through; do not create that collision (see the naming rule in
[§2](#2-naming-rules)).

### The `MovedModuleWarning` gate

`tests/conftest.py` turns `MovedModuleWarning` into an **error for the whole suite**, by appending
`error::zotero_mcp._shim.MovedModuleWarning` to pytest's `filterwarnings` list in `pytest_configure` (which
puts it after, and so ahead of, pyproject's `ignore::DeprecationWarning`). Nothing in this repo may reach a
shimmed path, so the suite fails closed: any access — including a function-level lazy import that no static
guard can see — raises. `ZOTERO_MCP_ALLOW_SHIM_PATHS=1` puts it back to a plain warning.

**Do not arm this with the interpreter's `-W` flag.** `python -W error::zotero_mcp._shim.MovedModuleWarning -m
pytest` looks like it works and does nothing: Python resolves the category at startup by importing
`zotero_mcp._shim` and binds *that* class object into `warnings.filters`, `conftest` then purges `zotero_mcp*`
from `sys.modules` (so the suite tests this checkout, not an editable install), and the class the re-imported
module defines is a different object the startup filter cannot match — after which `ignore::DeprecationWarning`
swallows the warning and the suite goes green having tested nothing. The same flag is fine in a plain
interpreter, where nothing purges the module, which is how a downstream caller would use it.
`tests/test_shim.py::test_the_suites_own_configuration_turns_a_shim_access_into_an_error` accesses a real
forwarded name under the suite's ambient filters and asserts it raises, so the gate cannot go inert unnoticed.

### Shim table

One row per moved module. Columns: **old import path**, **new import path**, **first release that warns**,
**release that removes the shim**. Append a row here and to `SHIMS` in `tests/test_module_layout.py` in the
same commit as the move, and add the matching `### Architecture` line to the CHANGELOG.

| Old import path | New import path | Deprecated in | Removed in |
|---|---|---|---|
| _(none yet — no module has moved)_ | | | |

All shims are removed together in 0.14.0.

## 6. The import-cost budget

`zotero_mcp/__init__.py` exposes `mcp` lazily because `from .server import mcp` at module scope made *every*
import of *any* submodule pay for FastMCP, the MCP SDK, pydantic, pyzotero, bibtexparser and unidecode. On
0.9.1 that made `from zotero_mcp.schema import valid_fields` — a stdlib-only, offline field lookup whose own
code runs in microseconds — cost roughly **1.45 s and ~1480 modules**. `tests/test_lightweight_imports.py`
pins the behaviour. Every assertion *about cost* runs its import in a **subprocess**, because import cost is
only observable on a cold interpreter: by the time the suite is running, `conftest` has already imported half
the package. The assertions about the lazy attribute still resolving run in-process, where a subprocess would
prove nothing. It checks that:

- `from zotero_mcp.schema import valid_fields` and `from zotero_mcp.identifiers import normalize_doi` load
  none of `fastmcp`, `mcp`, `pydantic`, `pyzotero`, `bibtexparser`;
- `import zotero_mcp; zotero_mcp.__version__` loads none of them either;
- `from zotero_mcp import mcp` still resolves, and is the same object as `zotero_mcp.server.mcp` (this one
  and the `dir()` and `AttributeError` checks beside it run in-process — they are about transparency, not
  cost);
- the `#485` startup gates decide from config *before* importing `semantic_search` or ChromaDB, for a missing
  config file, an absent reranker block, a disabled reranker and an unparseable config alike;
- `config_light` can answer the reranker gate without importing chromadb or numpy.

### The budget

**The budget is on the module count, not on the milliseconds.** Module count is reproducible: `import
zotero_mcp.schema` loaded exactly 73 modules in five of five runs. Wall time is not — the same import
measured seven times on one machine ranged from 9,044 to 12,192 µs, a spread of about 35%. Treat the
timings below as an order of magnitude, never as a threshold, and never compare them against a number
measured on another machine or another Python.

Measured on 2026-09-20, CPython 3.11.11 on macOS (arm64), from this checkout with `PYTHONPATH=src`.
"Modules" is `len(sys.modules)` after the import in a fresh interpreter; a bare interpreter starts at 32.
Both columns move with the Python version — 3.11 and 3.13 do not vendor the same stdlib — so re-measure the
baseline on your own interpreter before reading anything into a difference.

| Import | Modules | Budget | Time (indicative) |
|---|---|---|---|
| `import zotero_mcp` | 59 | under 100 modules | ~7 ms |
| `from zotero_mcp.identifiers import normalize_doi` | 61 | under 100 modules | ~6 ms |
| `from zotero_mcp.schema import valid_fields` | 73 | under 100 modules | ~9-12 ms |
| `import zotero_mcp.config_light` | 81 | under 100 modules | ~10 ms |
| `import zotero_mcp.server` | 1496 | none — informational | ~650 ms |

`import zotero_mcp.server` is the cost the lazy root exists to avoid, not a target to hold; it is listed so
the gap is visible.

Every *new* subpackage `__init__.py` is empty or lazy and must load **zero** of chromadb, torch, fitz,
pydantic, fastmcp and pyzotero. `tools/` is the sole exception and stays that way.

### How to measure

```bash
cd <repo root>

# The number that is actually budgeted: modules loaded in a cold interpreter.
PYTHONPATH=src python -c "import sys, zotero_mcp.schema; print(len(sys.modules))"

# Cumulative import time, in microseconds, in the last column. Noisy; run it
# several times and read the range, not one sample:
PYTHONPATH=src python -X importtime -c "import zotero_mcp.schema" 2>&1 | tail -1

# A new subpackage's __init__ must load none of the heavy dependencies.
# Exits 0 and prints an empty `heavy=[]` on success, exits 1 and names them
# on a leak, exits 1 with a traceback if the module does not import at all:
PYTHONPATH=src python -c '
import sys
name = sys.argv[1]
__import__(name)
heavy = sorted({"chromadb", "torch", "fitz", "pydantic", "fastmcp", "pyzotero"}
               & {m.split(".")[0] for m in sys.modules})
print(f"{name}: {len(sys.modules)} modules, heavy={heavy}")
sys.exit(1 if heavy else 0)
' zotero_mcp.backends

# The full guard:
python -m pytest tests/test_lightweight_imports.py -q
```

The third command also prints the module count, and reproduces the budget table's figures exactly, so it can
stand in for the first one.

Compare the **module count** before and after a move: a behaviour-neutral move does not change it. Do not
compare the millisecond figure that way — its run-to-run spread is wider than any regression small enough to
argue about.

## 7. Moving a module: the checklist

Every move or split works through all of these. Each row is a way the move can change behaviour without
failing anything, and the check that catches it.

| # | What can go wrong silently | How to catch it |
|---|---|---|
| 1 | A function-level lazy import still names the old path (there are ~95 of them in `src/`). It resolves through the shim, so nothing fails until the shim is removed — and `pyproject.toml` filters the warning. | A plain `python -m pytest`: `tests/conftest.py` makes `MovedModuleWarning` an error, so the access fails the test that reaches it ([§5](#the-movedmodulewarning-gate) — do **not** use `python -W error::...`, which is inert under the suite). Statically: `test_no_source_module_imports_a_shim_path`, plus `grep -rn "<old path>" src` for a path built by string concatenation. |
| 2 | A `monkeypatch.setattr("<old path>.<name>", ...)` left behind. It lands on the shim, so the test exercises the real implementation or a stale fake, and passes for the wrong reason. | `test_no_test_patches_or_imports_a_shim_path`; count the new-path occurrences with `grep -c` and check the total against the old-path census taken before the move. |
| 3 | A `__file__`-relative lookup now sits at the wrong depth (`skill_install.py` resolving `skills/`, `utils.py` resolving the install root). Works in-tree, breaks from the wheel. | `tests/test_skill_install.py`, `tests/test_install_hint.py`; `uv build --wheel` then `unzip -l` the result; run `zotero-mcp install-skill --list-targets` from the installed tool, not the checkout. |
| 4 | A subpackage `__init__.py` imports something heavy, re-creating the 1.45 s regression and defeating the `#485` startup gates. | `pytest tests/test_lightweight_imports.py`; the heavy-dependency probe in [§6](#how-to-measure) must exit `0` with `heavy=[]` for every package but `tools`. |
| 5 | Subprocess `-m` targets and console-script entry points still name the old module — a stale editable install keeps resolving them through the shim, so nothing looks wrong locally. | Print `entry_points(group="console_scripts")` after reinstalling; `python -m zotero_mcp.cli version`; `tests/test_skill_install.py::TestCliWiring`, `tests/test_generic_batch_flags.py`. |
| 6 | Strings derived from module names drift: `getLogger(__name__)` in a moved module no longer matches a literal `"zotero_mcp.extract"` silencer elsewhere, or a `"zotero_mcp.semantic_search" in sys.modules` check stops being true. | `grep -rn 'getLogger("zotero_mcp' src` — every name it prints must exist as a real module; plus the quoted-path guard in `test_module_layout.py`. |
| 7 | isort reorders a rewritten import block so that a top-level import now precedes a side-effecting one, or closes an import cycle that only lazy imports were avoiding. | `ruff check --select I`; cold `python -c "import zotero_mcp.server"` and `python -m zotero_mcp.cli.standalone --help`; `python scripts/measure_context_cost.py \| shasum` (tool registration order). |
| 8 | A new `tests/<pkg>/__init__.py` makes `cli`, `tools` or `formatting` importable as a top-level package while `tests/` is on `sys.path`; or a moved test computes the repo root with `parents[1]` and points one level too high. | The `find_spec` line in [§4](#4-test-layout) must print `[]` *before* the directory is added; `grep -rn "parents\[1\]\|parent\.parent" tests/<pkg>/` must be empty. |

Two more, both about the mechanics rather than the behaviour:

- **hatchling ships an `__init__.py`-less directory as an implicit namespace package.** Tests pass in-tree,
  but ruff's first-party detection and `git diff -M` get confusing and the wheel's contents are not what you
  think. Before committing, check that no directory holding `.py` files lacks an `__init__.py`.
- **`git mv` plus edits in one commit can drop below git's 50% rename threshold** for a small module, and
  `git log --follow` loses the history. Hence the **two-commit rule**: commit 1 is *only* the `git mv`, at
  100% similarity; commit 2 rewrites imports and patch strings; commit 3 adds the shim, the `__init__.py`,
  the docs and the CHANGELOG. Never edit a moved file in the commit that moves it. Moved modules switch
  relative imports to absolute, and keep an alias at the import site
  (`from zotero_mcp.backends import api as _client`) so that function bodies do not change at all.

### Proving a move was behaviour-neutral

```bash
git diff -M --stat                      # shows renames, not delete+add
git diff -M --color-moved=dimmed-zebra  # moved code is unchanged

python -m pytest --ignore=tests/test_lifespan.py -q   # same test count as before

# Same module count as before (not the same milliseconds — see §6):
PYTHONPATH=src python -c "import sys, zotero_mcp.schema; print(len(sys.modules))"

# The CLI's public contract, byte-identical before and after. Run it through
# `python -m` against this checkout, not via the installed `zotero-cli`:
PYTHONPATH=src python -m zotero_mcp.cli_standalone --json-schema > /tmp/schema.txt
echo "exit=$?" && shasum /tmp/schema.txt
```

**Check that the schema command exited 0 before comparing hashes.** A console script can fail for reasons
that have nothing to do with the change — a stale or shadowed editable install is the usual one — and
`shasum` of empty input is always `da39a3ee5e6b4b0d3255bfef95601890afd80709`. Two matching hashes of two
failures look exactly like a pass. This applies to any before/after hash: a hash of a failure is not
evidence.

**Lint the files the change touches, not the tree.** `ruff check src tests` currently reports 119
pre-existing errors (44 `I001`, 37 `F401`, 29 `F841` and a tail of others), none of them from any one change:
pre-commit only ever lints staged files, so the tree has never been clean all at once. Use the version
`.pre-commit-config.yaml` pins, scoped to the diff:

```bash
git diff --name-only main...HEAD -- '*.py' | xargs -r uvx ruff@0.9.10 check
git diff --name-only main...HEAD -- '*.py' | xargs -r uvx ruff@0.9.10 format --check
```

Do not clear the tree-wide backlog inside a move PR. `--fix` rewrites import blocks in files the PR only
renamed, which pushes them below git's rename-detection threshold and destroys exactly the `git diff -M`
evidence the move is supposed to produce.
