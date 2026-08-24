"""Install the packaged ``zotero-cli`` agent skill into whatever harness is here.

The skill ships inside the wheel, but an agent only reads its own conventions,
and those differ: Claude Code reads ``SKILL.md`` files under a ``skills``
directory, Cursor and Windsurf read rule files with their own frontmatter, and
a growing set of tools (Codex, Amp, OpenCode, Jules, Gemini) read a single
project-level instructions file. One skill, several destinations.

``zotero-mcp install-skill`` with no arguments detects which of those are
present and writes to each. Detection beats a flag here because the answer is
already on disk — a repo with ``.cursor/`` and ``AGENTS.md`` wants both, and
asking the user to know that is asking them to learn our taxonomy.

Two rules hold for every target:

* **Never clobber.** A destination that exists and differs from what we would
  write is reported, not overwritten, unless ``force`` is passed. For the
  shared instruction files that is stricter still: we only ever manage the
  text between our own markers, so the rest of someone's ``AGENTS.md`` is
  untouchable by construction rather than by care.
* **Progressive disclosure survives.** Shared instruction files get a short
  pointer block, not the whole skill body; the body lands beside it in a file
  the agent reads only when it decides the skill is relevant. Pasting 1,400
  tokens into every agent's always-loaded context would give away exactly the
  advantage the CLI route exists for.
"""

from __future__ import annotations

import filecmp
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: Name of the skill, used as the directory/file stem at every destination.
SKILL_NAME = "zotero-cli"

#: Delimiters for the block we manage inside a shared instructions file. Both
#: carry the command that wrote them, so someone finding it a year later knows
#: what put it there and how to update it.
BLOCK_BEGIN = f"<!-- BEGIN {SKILL_NAME} skill (managed by `zotero-mcp install-skill`) -->"
BLOCK_END = f"<!-- END {SKILL_NAME} skill -->"


# ---------------------------------------------------------------------------
# The packaged skill
# ---------------------------------------------------------------------------

def packaged_skill_dir() -> Path:
    """Where the skill lives inside the installed package."""
    return Path(__file__).resolve().parent / "skills" / SKILL_NAME


def skill_description() -> str:
    """The skill's own one-line description, from its frontmatter.

    Read rather than restated so the pointer blocks and the SKILL.md can never
    describe the skill differently.
    """
    text = (packaged_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    match = re.search(r"^description:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1) if match else "Read and write a Zotero library from the shell."


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    """What happened for one harness."""

    target: str
    status: str  # "installed" | "current" | "skipped" | "error"
    detail: str
    paths: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("installed", "current")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _copy_skill_dir(dst: Path, force: bool) -> TargetResult | None:
    """Copy the packaged directory to *dst*. Returns None when it went fine."""
    src = packaged_skill_dir()
    if dst.exists():
        differences = _describe_difference(src, dst)
        if not differences:
            return TargetResult("", "current", f"already up to date: {dst}", [dst])
        if not force:
            listing = "; ".join(differences[:4])
            return TargetResult(
                "", "skipped",
                f"{dst} exists and differs ({listing}). Re-run with --force to "
                f"overwrite; --force keeps no backup.",
                [dst],
            )
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return None


def _describe_difference(src: Path, dst: Path) -> list[str]:
    changed: list[str] = []
    for src_file in sorted(src.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        if not dst_file.exists():
            changed.append(f"missing {rel}")
        elif not filecmp.cmp(src_file, dst_file, shallow=False):
            changed.append(f"differs {rel}")
    for dst_file in sorted(dst.rglob("*")):
        if dst_file.is_file() and not (src / dst_file.relative_to(dst)).exists():
            changed.append(f"extra {dst_file.relative_to(dst)}")
    return changed


def _display_path(path: Path, root: Path) -> str:
    """A path as it should appear inside a markdown document.

    Always forward slashes. ``Path.relative_to`` hands back an OS-native path,
    so on Windows this text would otherwise read
    ``.agents\\skills\\zotero-cli\\SKILL.md`` — wrong for a markdown
    document, and a backslash is an escape character in enough contexts that
    an agent can misread the path it is told to open.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _skill_body(*, reference_path: Path, root: Path) -> str:
    """The SKILL.md body, with its pointer to the command reference resolved.

    The packaged body says "reference.md in this skill directory", which is
    true where the whole directory is copied intact (Claude) and false
    everywhere else. An agent told to read a file that is not there does not
    fall back gracefully; it reports the library as broken.
    """
    src = (packaged_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    parts = src.split("---", 2)
    body = parts[2].lstrip("\n") if len(parts) > 2 else src
    return body.replace(
        "`reference.md` in this skill directory",
        f"`{_display_path(reference_path, root)}`",
    )


def _pointer_block(skill_path: Path, root: Path) -> str:
    """The short block inserted into a shared instructions file.

    Deliberately not the skill body. This text is in the agent's context for
    every turn of every conversation, so it says only what is needed to decide
    whether to open the real thing.
    """
    rel = _display_path(skill_path, root)
    return "\n".join([
        BLOCK_BEGIN,
        "",
        "## Zotero library access",
        "",
        skill_description(),
        "",
        f"Read `{rel}` before using it. Prefer it over the Zotero MCP server "
        f"when you have shell access: the MCP tool schemas cost ~13k tokens of "
        f"context on every request, this costs only what you run.",
        "",
        "Quick check that it is set up: `zotero-cli config`. "
        "Pass `--json` whenever you will parse the output.",
        "",
        BLOCK_END,
    ])


def _upsert_block(path: Path, block: str, force: bool) -> TargetResult | None:
    """Insert or update our block in *path*, leaving everything else alone.

    Creates the file when it does not exist. When it does, the block is
    replaced in place if present and appended if not — so re-running is safe
    and never duplicates, and a file the user has written around our markers
    keeps every line outside them.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block + "\n", encoding="utf-8")
        return None

    existing = path.read_text(encoding="utf-8")
    if BLOCK_BEGIN in existing and BLOCK_END in existing:
        start = existing.index(BLOCK_BEGIN)
        end = existing.index(BLOCK_END) + len(BLOCK_END)
        if existing[start:end] == block:
            return TargetResult("", "current", f"already up to date: {path}", [path])
        if not force:
            return TargetResult(
                "", "skipped",
                f"{path} has a {SKILL_NAME} block that differs from the packaged "
                f"one. Re-run with --force to replace it (only the block between "
                f"the markers changes).",
                [path],
            )
        path.write_text(existing[:start] + block + existing[end:], encoding="utf-8")
        return None

    separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    path.write_text(existing + separator + block + "\n", encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def _install_claude(root: Path, force: bool, *, user_scope: bool) -> TargetResult:
    base = (Path.home() / ".claude") if user_scope else (root / ".claude")
    dst = base / "skills" / SKILL_NAME
    problem = _copy_skill_dir(dst, force)
    name = "claude-user" if user_scope else "claude"
    if problem:
        return TargetResult(name, problem.status, problem.detail, problem.paths)
    return TargetResult(name, "installed", f"skill directory at {dst}",
                        [dst / "SKILL.md", dst / "reference.md"])


def _install_rules_file(root: Path, force: bool, *, target: str, rel_dir: str,
                        suffix: str, frontmatter: str) -> TargetResult:
    """Cursor/Windsurf style: one rule file with harness-specific frontmatter.

    The body is the skill's own body, so there is one source of truth; only
    the frontmatter differs per harness.
    """
    dst = root / rel_dir / f"{SKILL_NAME}{suffix}"

    # The command reference does NOT go in the rules directory. A harness that
    # loads every file it finds there would pull 3k tokens of flag listings
    # into context permanently, which is the opposite of what a
    # `alwaysApply: false` rule is for. It lives with the shared copy instead,
    # and the body is rewritten to point at where it actually is.
    skill_dir = root / ".agents" / "skills" / SKILL_NAME
    problem = _copy_skill_dir(skill_dir, force)
    if problem and problem.status == "skipped":
        return TargetResult(target, problem.status, problem.detail, problem.paths)

    body = _skill_body(reference_path=(skill_dir / "reference.md"), root=root)
    content = frontmatter + "\n" + body

    if dst.exists() and dst.read_text(encoding="utf-8") == content:
        return TargetResult(target, "current", f"already up to date: {dst}", [dst])
    if dst.exists() and not force:
        return TargetResult(
            target, "skipped",
            f"{dst} exists and differs. Re-run with --force to overwrite.",
            [dst],
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    return TargetResult(target, "installed", f"rule file at {dst}",
                        [dst, skill_dir / "reference.md"])


def _install_instructions_file(root: Path, force: bool, *, target: str,
                               filename: str) -> TargetResult:
    """AGENTS.md / GEMINI.md style: a short pointer block plus the real skill.

    The skill body goes to `.agents/skills/<name>/`, which no harness owns, so
    several instruction files can point at one copy.
    """
    skill_dir = root / ".agents" / "skills" / SKILL_NAME
    problem = _copy_skill_dir(skill_dir, force)
    if problem and problem.status == "skipped":
        return TargetResult(target, problem.status, problem.detail, problem.paths)

    path = root / filename
    block = _pointer_block(skill_dir / "SKILL.md", root)
    problem = _upsert_block(path, block, force)
    if problem and problem.status == "skipped":
        return TargetResult(target, problem.status, problem.detail, problem.paths)
    if problem and problem.status == "current":
        return TargetResult(target, "current", f"already up to date: {path}", [path])
    return TargetResult(target, "installed", f"pointer in {path}, skill in {skill_dir}",
                        [path, skill_dir / "SKILL.md"])


#: Every supported harness: how to detect it, and how to install into it.
#:
#: `detect` answers "is this harness in use here?" from what is on disk. It is
#: deliberately generous — a directory or file the harness owns is enough,
#: because a false positive costs one unused file and a false negative costs
#: the user the whole feature.
TARGETS: dict[str, dict] = {
    "claude": {
        "label": "Claude Code (project)",
        "detect": lambda root: (root / ".claude").is_dir(),
        "install": lambda root, force: _install_claude(root, force, user_scope=False),
    },
    "claude-user": {
        "label": "Claude Code (user, ~/.claude)",
        "detect": lambda root: (Path.home() / ".claude").is_dir(),
        "install": lambda root, force: _install_claude(root, force, user_scope=True),
    },
    "cursor": {
        "label": "Cursor",
        "detect": lambda root: (root / ".cursor").is_dir(),
        "install": lambda root, force: _install_rules_file(
            root, force, target="cursor", rel_dir=".cursor/rules", suffix=".mdc",
            frontmatter=(
                "---\n"
                f"description: {skill_description()}\n"
                "alwaysApply: false\n"
                "---\n"
            ),
        ),
    },
    "windsurf": {
        "label": "Windsurf",
        "detect": lambda root: (root / ".windsurf").is_dir(),
        "install": lambda root, force: _install_rules_file(
            root, force, target="windsurf", rel_dir=".windsurf/rules", suffix=".md",
            frontmatter=(
                "---\n"
                "trigger: model_decision\n"
                f"description: {skill_description()}\n"
                "---\n"
            ),
        ),
    },
    "agents": {
        "label": "AGENTS.md (Codex, Amp, OpenCode, Jules, ...)",
        "detect": lambda root: (root / "AGENTS.md").is_file(),
        "install": lambda root, force: _install_instructions_file(
            root, force, target="agents", filename="AGENTS.md"),
    },
    "gemini": {
        "label": "Gemini CLI",
        "detect": lambda root: (root / "GEMINI.md").is_file() or (root / ".gemini").is_dir(),
        "install": lambda root, force: _install_instructions_file(
            root, force, target="gemini", filename="GEMINI.md"),
    },
}


def detect_targets(root: Path | None = None) -> list[str]:
    """Which harnesses are in use in *root* (default: the current directory)."""
    root = Path(root) if root else Path.cwd()
    return [name for name, spec in TARGETS.items() if spec["detect"](root)]


def install_skill(
    targets: list[str] | None = None,
    *,
    root: Path | None = None,
    force: bool = False,
) -> list[TargetResult]:
    """Install the skill into *targets*, or into every detected harness.

    Returns one :class:`TargetResult` per target attempted. An empty list means
    nothing was detected, which the caller reports as guidance rather than as
    an error — the user may simply not have opened this directory in an agent
    yet.
    """
    root = Path(root) if root else Path.cwd()

    if not packaged_skill_dir().is_dir():
        return [TargetResult(
            "-", "error",
            f"The packaged skill is missing from this installation (looked in "
            f"{packaged_skill_dir()}). Reinstall zotero-mcp-server.",
        )]

    if targets:
        unknown = [t for t in targets if t not in TARGETS]
        if unknown:
            return [TargetResult(
                "-", "error",
                f"Unknown target(s): {', '.join(unknown)}. "
                f"Known: {', '.join(TARGETS)}.",
            )]
        chosen = targets
    else:
        chosen = detect_targets(root)

    results = []
    for name in chosen:
        try:
            results.append(TARGETS[name]["install"](root, force))
        except Exception as exc:  # one broken target must not stop the rest
            results.append(TargetResult(name, "error", f"{type(exc).__name__}: {exc}"))
    return results


def format_results(results: list[TargetResult], root: Path | None = None) -> str:
    """Human-readable summary of what an install run did."""
    root = Path(root) if root else Path.cwd()
    if not results:
        detected = ", ".join(TARGETS)
        return (
            "No agent harness detected in this directory.\n\n"
            "install-skill looks for .claude/, .cursor/, .windsurf/, AGENTS.md, "
            "GEMINI.md, or ~/.claude/.\n"
            f"Name one explicitly to install anyway: "
            f"zotero-mcp install-skill --target <{'|'.join(TARGETS)}>\n"
            f"(known targets: {detected})"
        )

    lines = []
    for result in results:
        marker = {"installed": "+", "current": "=", "skipped": "!", "error": "x"}[result.status]
        label = TARGETS.get(result.target, {}).get("label", result.target)
        lines.append(f"  {marker} {label}: {result.detail}")

    installed = sum(1 for r in results if r.status == "installed")
    header = (
        f"Installed the {SKILL_NAME} skill into {installed} target(s):"
        if installed else "Nothing to do:"
    )
    footer = ""
    if any(r.status == "installed" for r in results):
        footer = "\n\nRestart your agent session to pick it up."
    return header + "\n" + "\n".join(lines) + footer
