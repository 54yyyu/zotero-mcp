"""`zotero-mcp install-skill` reaches whatever agent harness is set up here.

The skill ships in the wheel, but every harness reads a different convention:
Claude Code wants a SKILL.md under a skills directory, Cursor and Windsurf
want rule files with their own frontmatter, and a growing set of tools read a
single project-level instructions file. One command has to cover all of them
or the CLI route is only available to whoever happens to use Claude Code.

The failures worth guarding are not failed copies. They are a successful copy
that overwrote someone's customisation, a second run that duplicated a block
in AGENTS.md, and a rule file that points at a reference.md which is not
there -- an agent told to read a missing file reports the library as broken
rather than degrading gracefully.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from zotero_mcp.skill_install import (
    BLOCK_BEGIN,
    BLOCK_END,
    SKILL_NAME,
    TARGETS,
    detect_targets,
    format_results,
    install_skill,
    packaged_skill_dir,
    skill_description,
)


@pytest.fixture(autouse=True)
def fake_home(tmp_path_factory, monkeypatch):
    """Point Path.home() at a scratch directory for every test in this file.

    The `claude-user` target writes to ~/.claude/skills, and auto-detect finds
    it whenever that directory exists — which it does on any machine that has
    ever run Claude Code, including a maintainer's. Without this, running the
    suite silently installs into the developer's real home directory. It did,
    once, which is how this fixture came to exist.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture
def project(tmp_path):
    """An empty project directory standing in for the user's repo."""
    return tmp_path


class TestPackagedSkill:
    def test_ships_with_the_package(self):
        src = packaged_skill_dir()
        assert src.is_dir(), f"packaged skill missing at {src}"
        assert (src / "SKILL.md").is_file()
        assert (src / "reference.md").is_file()

    def test_lives_inside_the_importable_package(self):
        """Anywhere else and the wheel build would not include it."""
        import zotero_mcp

        assert packaged_skill_dir().is_relative_to(
            Path(zotero_mcp.__file__).resolve().parent
        )

    def test_description_comes_from_the_skill_itself(self):
        """Restating it would let the pointer blocks and SKILL.md drift."""
        described = skill_description()
        assert described
        assert described in (packaged_skill_dir() / "SKILL.md").read_text(encoding="utf-8")


class TestDetection:
    @pytest.mark.parametrize("marker,expected", [
        (".claude", "claude"),
        (".cursor", "cursor"),
        (".windsurf", "windsurf"),
    ])
    def test_directory_markers(self, project, marker, expected):
        (project / marker).mkdir()
        assert expected in detect_targets(project)

    @pytest.mark.parametrize("marker,expected", [
        ("AGENTS.md", "agents"),
        ("GEMINI.md", "gemini"),
    ])
    def test_file_markers(self, project, marker, expected):
        (project / marker).write_text("# hi", encoding="utf-8")
        assert expected in detect_targets(project)

    def test_several_at_once(self, project):
        """A repo with Cursor and AGENTS.md wants both, and the user should not
        have to know that."""
        (project / ".cursor").mkdir()
        (project / "AGENTS.md").write_text("# hi", encoding="utf-8")
        detected = detect_targets(project)
        assert "cursor" in detected and "agents" in detected

    def test_empty_project_detects_nothing(self, project):
        assert detect_targets(project) == []

    def test_claude_user_is_detected_from_the_home_directory(self, fake_home, project):
        """It is a property of the machine, not of this directory."""
        (fake_home / ".claude").mkdir()
        assert "claude-user" in detect_targets(project)


class TestClaude:
    def test_installs_a_self_contained_skill_directory(self, project):
        (project / ".claude").mkdir()
        results = install_skill(["claude"], root=project)
        dst = project / ".claude" / "skills" / SKILL_NAME
        assert results[0].status == "installed"
        assert (dst / "SKILL.md").is_file()
        # Claude gets the directory intact, so the body's own "reference.md in
        # this skill directory" wording is true here.
        assert (dst / "reference.md").is_file()

    def test_refuses_to_overwrite_an_edited_skill(self, project):
        (project / ".claude").mkdir()
        install_skill(["claude"], root=project)
        edited = project / ".claude" / "skills" / SKILL_NAME / "SKILL.md"
        edited.write_text("my own version", encoding="utf-8")

        results = install_skill(["claude"], root=project)

        assert results[0].status == "skipped"
        assert edited.read_text(encoding="utf-8") == "my own version"

    def test_force_overwrites(self, project):
        (project / ".claude").mkdir()
        install_skill(["claude"], root=project)
        edited = project / ".claude" / "skills" / SKILL_NAME / "SKILL.md"
        edited.write_text("my own version", encoding="utf-8")

        install_skill(["claude"], root=project, force=True)

        assert edited.read_text(encoding="utf-8") != "my own version"


class TestRuleFileHarnesses:
    @pytest.mark.parametrize("target,path,marker", [
        ("cursor", ".cursor/rules/zotero-cli.mdc", "alwaysApply: false"),
        ("windsurf", ".windsurf/rules/zotero-cli.md", "trigger: model_decision"),
    ])
    def test_writes_a_rule_file_with_the_harness_frontmatter(
        self, project, target, path, marker
    ):
        results = install_skill([target], root=project)
        rule = project / path
        assert results[0].status == "installed"
        assert rule.is_file()
        text = rule.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert marker in text
        assert skill_description() in text

    def test_reference_is_not_written_into_the_rules_directory(self, project):
        """A harness that loads everything in rules/ would pull 3k tokens of
        flag listings permanently into context -- the opposite of what an
        on-demand rule is for."""
        install_skill(["cursor"], root=project)
        rules_dir = project / ".cursor" / "rules"
        assert not (rules_dir / "reference.md").exists()
        assert [p.name for p in rules_dir.iterdir()] == ["zotero-cli.mdc"]

    def test_rule_body_points_at_a_reference_that_exists(self, project):
        """The packaged body says "reference.md in this skill directory",
        which is false once the file is copied somewhere else."""
        install_skill(["cursor"], root=project)
        text = (project / ".cursor" / "rules" / "zotero-cli.mdc").read_text(encoding="utf-8")
        assert "in this skill directory" not in text
        assert ".agents/skills/zotero-cli/reference.md" in text
        assert (project / ".agents" / "skills" / SKILL_NAME / "reference.md").is_file()

    def test_reinstall_is_a_no_op(self, project):
        install_skill(["cursor"], root=project)
        results = install_skill(["cursor"], root=project)
        assert results[0].status == "current"


class TestInstructionsFileHarnesses:
    def test_creates_the_file_when_absent(self, project):
        results = install_skill(["agents"], root=project)
        agents = project / "AGENTS.md"
        assert results[0].status == "installed"
        assert BLOCK_BEGIN in agents.read_text(encoding="utf-8")

    def test_existing_content_is_preserved(self, project):
        agents = project / "AGENTS.md"
        agents.write_text("# My project\n\nMy own instructions.\n", encoding="utf-8")

        install_skill(["agents"], root=project)

        text = agents.read_text(encoding="utf-8")
        assert "# My project" in text
        assert "My own instructions." in text
        assert BLOCK_BEGIN in text

    def test_the_block_is_a_pointer_not_the_whole_skill(self, project):
        """This text sits in context on every turn. Pasting the body here
        would spend the advantage the CLI route exists for."""
        install_skill(["agents"], root=project)
        block = _block_of(project / "AGENTS.md")
        body = (packaged_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
        assert len(block) < len(body) / 3
        assert ".agents/skills/zotero-cli/SKILL.md" in block

    def test_the_real_skill_lands_beside_it(self, project):
        install_skill(["agents"], root=project)
        assert (project / ".agents" / "skills" / SKILL_NAME / "SKILL.md").is_file()

    def test_reinstall_does_not_duplicate_the_block(self, project):
        install_skill(["agents"], root=project)
        install_skill(["agents"], root=project)
        text = (project / "AGENTS.md").read_text(encoding="utf-8")
        assert text.count(BLOCK_BEGIN) == 1
        assert text.count(BLOCK_END) == 1

    def test_an_edited_block_is_not_replaced_without_force(self, project):
        install_skill(["agents"], root=project)
        agents = project / "AGENTS.md"
        agents.write_text(
            agents.read_text(encoding="utf-8").replace("Quick check", "MY EDIT"),
            encoding="utf-8",
        )

        results = install_skill(["agents"], root=project)

        assert results[0].status == "skipped"
        assert "MY EDIT" in agents.read_text(encoding="utf-8")

    def test_force_replaces_only_the_managed_block(self, project):
        agents = project / "AGENTS.md"
        agents.write_text("# Mine\n\nKeep me.\n", encoding="utf-8")
        install_skill(["agents"], root=project)
        agents.write_text(
            agents.read_text(encoding="utf-8").replace("Quick check", "MY EDIT"),
            encoding="utf-8",
        )

        install_skill(["agents"], root=project, force=True)

        text = agents.read_text(encoding="utf-8")
        assert "Keep me." in text          # their prose survives --force
        assert "MY EDIT" not in text       # our block is replaced
        assert text.count(BLOCK_BEGIN) == 1

    def test_gemini_uses_its_own_file(self, project):
        install_skill(["gemini"], root=project)
        assert BLOCK_BEGIN in (project / "GEMINI.md").read_text(encoding="utf-8")
        assert not (project / "AGENTS.md").exists()

    def test_two_instruction_files_share_one_skill_copy(self, project):
        """Nothing owns .agents/, which is why both can point at it."""
        install_skill(["agents", "gemini"], root=project)
        assert BLOCK_BEGIN in (project / "AGENTS.md").read_text(encoding="utf-8")
        assert BLOCK_BEGIN in (project / "GEMINI.md").read_text(encoding="utf-8")
        copies = list((project / ".agents" / "skills").iterdir())
        assert len(copies) == 1


def _block_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text[text.index(BLOCK_BEGIN):text.index(BLOCK_END) + len(BLOCK_END)]


class TestAutoDetectInstall:
    def test_installs_into_every_detected_harness(self, project):
        (project / ".cursor").mkdir()
        (project / "AGENTS.md").write_text("# hi", encoding="utf-8")

        results = install_skill(root=project)
        targets = {r.target for r in results}

        assert "cursor" in targets and "agents" in targets
        assert (project / ".cursor" / "rules" / "zotero-cli.mdc").is_file()
        assert BLOCK_BEGIN in (project / "AGENTS.md").read_text(encoding="utf-8")

    def test_unknown_target_is_an_error_not_a_silent_no_op(self, project):
        results = install_skill(["notaharness"], root=project)
        assert results[0].status == "error"
        assert "notaharness" in results[0].detail

    def test_one_broken_target_does_not_stop_the_others(self, project, monkeypatch):
        (project / ".cursor").mkdir()

        def explode(root, force):
            raise RuntimeError("boom")

        monkeypatch.setitem(TARGETS["cursor"], "install", explode)
        results = install_skill(["cursor", "agents"], root=project)

        assert {r.status for r in results} == {"error", "installed"}
        assert BLOCK_BEGIN in (project / "AGENTS.md").read_text(encoding="utf-8")

    def test_no_harness_detected_gives_guidance_not_an_error(self, project):
        results = install_skill(root=project)
        message = format_results(results, project)
        assert results == []
        assert "No agent harness detected" in message
        assert "--target" in message


class TestCliWiring:
    def _run(self, *args, cwd=None, home=None):
        # monkeypatch cannot reach a child process, so the fake home is passed
        # through the environment instead. Without it these would install into
        # the real ~/.claude.
        env = dict(os.environ)
        if home is not None:
            env["HOME"] = str(home)
            env["USERPROFILE"] = str(home)  # Windows
        return subprocess.run(
            [sys.executable, "-m", "zotero_mcp.cli", "install-skill", *args],
            capture_output=True, text=True, timeout=120, cwd=cwd, env=env,
        )

    def test_help_lists_the_new_flags(self):
        result = self._run("--help")
        assert result.returncode == 0, result.stderr
        for flag in ("--target", "--list-targets", "--force", "--root"):
            assert flag in result.stdout

    def test_list_targets_names_every_harness(self, project, fake_home):
        result = self._run("--list-targets", "--root", str(project), home=fake_home)
        assert result.returncode == 0, result.stderr
        for name in TARGETS:
            assert name in result.stdout

    def test_installing_via_the_cli_writes_files(self, project, fake_home):
        (project / ".cursor").mkdir()
        result = self._run("--root", str(project), home=fake_home)
        assert result.returncode == 0, result.stderr
        assert (project / ".cursor" / "rules" / "zotero-cli.mdc").is_file()

    def test_a_wholly_refused_install_exits_non_zero(self, project, fake_home):
        """A provisioning script has to be able to tell it did not happen."""
        install_skill(["agents"], root=project)
        agents = project / "AGENTS.md"
        agents.write_text(
            agents.read_text(encoding="utf-8").replace("Quick check", "EDIT"),
            encoding="utf-8",
        )
        result = self._run("--target", "agents", "--root", str(project), home=fake_home)
        assert result.returncode == 1
