"""Install the packaged ``zotero-cli`` agent skill into a skills directory.

The skill ships inside the wheel so that ``pip install zotero-mcp-server``
carries it, but an agent only finds a skill in its own skills directory. This
copies one to the other.

Kept deliberately dumb -- copy a directory, report what happened -- because
the interesting failure is not the copy, it is overwriting an edited skill
without saying so. Installing over an existing directory therefore requires
``force`` and reports the difference either way.
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

#: Name of the skill directory, both in the package and once installed.
SKILL_NAME = "zotero-cli"


def packaged_skill_dir() -> Path:
    """Where the skill lives inside the installed package."""
    return Path(__file__).resolve().parent / "skills" / SKILL_NAME


def default_skills_root(scope: str = "user") -> Path:
    """The skills directory an agent reads.

    ``user`` is ``~/.claude/skills`` (available in every project);
    ``project`` is ``.claude/skills`` under the current directory (shared with
    a repo, and the right choice when the library is part of a project's
    workflow rather than the person's).
    """
    if scope == "project":
        return Path.cwd() / ".claude" / "skills"
    return Path.home() / ".claude" / "skills"


def _describe_difference(src: Path, dst: Path) -> list[str]:
    """Files that differ between the packaged skill and an installed one."""
    changed: list[str] = []
    for src_file in sorted(src.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        if not dst_file.exists():
            changed.append(f"new: {rel}")
        elif not filecmp.cmp(src_file, dst_file, shallow=False):
            changed.append(f"differs: {rel}")
    for dst_file in sorted(dst.rglob("*")):
        if dst_file.is_file():
            rel = dst_file.relative_to(dst)
            if not (src / rel).exists():
                changed.append(f"only installed: {rel}")
    return changed


def install_skill(
    skills_root: Path | None = None,
    *,
    scope: str = "user",
    force: bool = False,
) -> tuple[bool, str]:
    """Copy the packaged skill into *skills_root*.

    Returns ``(installed, message)``. ``installed`` is False when the copy was
    declined -- an existing directory without ``force`` -- which is a refusal
    to act, not an error.
    """
    src = packaged_skill_dir()
    if not src.is_dir():
        return False, (
            f"The packaged skill is missing from this installation "
            f"(looked in {src}). Reinstall zotero-mcp-server."
        )

    root = Path(skills_root) if skills_root else default_skills_root(scope)
    dst = root / SKILL_NAME

    if dst.exists():
        differences = _describe_difference(src, dst)
        if not differences:
            return True, f"Already up to date: {dst}"
        if not force:
            listing = "\n".join(f"  {d}" for d in differences[:10])
            more = (
                f"\n  ... and {len(differences) - 10} more"
                if len(differences) > 10 else ""
            )
            return False, (
                f"A skill already exists at {dst} and differs from the packaged "
                f"one:\n{listing}{more}\n"
                f"Re-run with --force to overwrite it. If you edited it, copy it "
                f"somewhere else first -- --force does not keep a backup."
            )
        shutil.rmtree(dst)

    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    files = sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())
    listing = "\n".join(f"  {f}" for f in files)
    return True, f"Installed the zotero-cli skill to {dst}:\n{listing}"
