"""The packaged skill stays true to the CLI it documents.

A skill that names a flag the CLI does not accept is worse than one that omits
it: an agent reading it will confidently run a command that fails, and the
failure looks like a broken library rather than stale documentation. The
reference is generated from `build_parser()` for that reason, and this test
fails when the checked-in copy no longer matches -- which is the moment
someone adds a CLI flag and forgets to regenerate.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO / "src" / "zotero_mcp" / "skills" / "zotero-cli"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCE = SKILL_DIR / "reference.md"
GENERATOR = REPO / "scripts" / "gen_skill_reference.py"


def test_reference_matches_the_current_parser():
    """Regenerate in memory and compare. If this fails, run
    `python scripts/gen_skill_reference.py` and commit the result."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_skill_reference", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert REFERENCE.read_text(encoding="utf-8") == module.build(), (
        "src/zotero_mcp/skills/zotero-cli/reference.md is stale -- "
        "run `python scripts/gen_skill_reference.py` and commit the result."
    )


class TestSkillFrontmatter:
    def test_has_name_and_description(self):
        text = SKILL_MD.read_text(encoding="utf-8")
        assert text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
        frontmatter = text.split("---", 2)[1]
        assert re.search(r"^name:\s*zotero-cli\s*$", frontmatter, re.MULTILINE)
        assert re.search(r"^description:\s*\S", frontmatter, re.MULTILINE)

    def test_description_names_what_triggers_it(self):
        """The description is the only part loaded until the skill fires, so
        it has to carry the words a user would actually say."""
        text = SKILL_MD.read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1].lower()
        for trigger in ("zotero", "librar", "paper", "citation"):
            assert trigger in frontmatter, f"description should mention {trigger!r}"


class TestSkillAccuracy:
    """Every command and flag the skill body shows must actually exist."""

    @staticmethod
    def _cli_tokens():
        import argparse

        from zotero_mcp.cli_standalone import build_parser

        parser = build_parser()
        commands: set[str] = set()
        flags: set[str] = {"--json", "--json-schema", "-v", "--verbose"}

        def walk(p):
            for action in p._actions:
                flags.update(action.option_strings)
                if isinstance(action, argparse._SubParsersAction):
                    for name, sub in action.choices.items():
                        commands.add(name)
                        walk(sub)

        walk(parser)
        return commands, flags

    def test_every_flag_shown_in_the_skill_exists(self):
        commands, flags = self._cli_tokens()
        body = SKILL_MD.read_text(encoding="utf-8")

        shown = set()
        for block in re.findall(r"```bash\n(.*?)```", body, re.DOTALL):
            for line in block.splitlines():
                if "zotero-cli" not in line:
                    continue
                shown.update(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", line))
        # Also catch flags named in the prose tables.
        shown.update(re.findall(r"`--[a-z][a-z-]*", body))
        shown = {s.lstrip("`") for s in shown}

        unknown = sorted(shown - flags)
        assert not unknown, f"SKILL.md shows flags the CLI does not accept: {unknown}"

    def test_every_command_shown_in_the_skill_exists(self):
        commands, _ = self._cli_tokens()
        body = SKILL_MD.read_text(encoding="utf-8")

        shown = set()
        for block in re.findall(r"```bash\n(.*?)```", body, re.DOTALL):
            for line in block.splitlines():
                match = re.search(r"zotero-cli\s+((?:--\S+\s+)*)([a-z-]+)", line)
                if match:
                    shown.add(match.group(2))
        shown -= {"config"}  # top-level, always present
        unknown = sorted(s for s in shown if s not in commands and s != "config")
        assert not unknown, f"SKILL.md shows commands that do not exist: {unknown}"


def test_generator_is_executable_and_idempotent(tmp_path):
    """Running it twice must not produce a diff -- otherwise the staleness
    test above would fail on every second commit.

    Generated into tmp_path, never over the checked-in copy: writing to the
    tracked file left the working tree dirty after every test run, and on
    Windows it rewrote the whole file in CRLF as it went.
    """
    out = tmp_path / "reference.md"

    first = subprocess.run(
        [sys.executable, str(GENERATOR), str(out)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert first.returncode == 0, first.stderr
    after_first = out.read_bytes()

    second = subprocess.run(
        [sys.executable, str(GENERATOR), str(out)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert second.returncode == 0, second.stderr
    assert out.read_bytes() == after_first


def test_generator_writes_lf_line_endings(tmp_path):
    """Compared on bytes, deliberately: read_text() turns CRLF into LF on the
    way in, so a text comparison can never see the drift it is meant to catch.
    """
    out = tmp_path / "reference.md"
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), str(out)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert b"\r\n" not in out.read_bytes()


def test_checked_in_reference_uses_lf_line_endings():
    """The tracked copy is LF like the rest of the repo."""
    assert b"\r\n" not in REFERENCE.read_bytes()
