"""`zotero-mcp install-skill` copies the packaged skill without losing edits.

The skill ships in the wheel, but an agent only reads its own skills
directory. The interesting failure here is not a failed copy -- it is a
successful one that silently overwrote a skill the user had customised, which
is why installing over a modified directory is refused rather than done.
"""

from pathlib import Path

from zotero_mcp.skill_install import (
    SKILL_NAME,
    default_skills_root,
    install_skill,
    packaged_skill_dir,
)


class TestPackagedSkill:
    def test_ships_with_the_package(self):
        """If this fails the wheel is missing the skill and install-skill
        cannot work for anyone who installed from PyPI."""
        src = packaged_skill_dir()
        assert src.is_dir(), f"packaged skill missing at {src}"
        assert (src / "SKILL.md").is_file()
        assert (src / "reference.md").is_file()

    def test_lives_inside_the_importable_package(self):
        """Anywhere else and it would not be included by the wheel build,
        which packages `src/zotero_mcp` and nothing above it."""
        import zotero_mcp

        package_root = Path(zotero_mcp.__file__).resolve().parent
        assert packaged_skill_dir().is_relative_to(package_root)


class TestDefaultRoots:
    def test_user_scope_is_the_home_skills_directory(self):
        assert default_skills_root("user") == Path.home() / ".claude" / "skills"

    def test_project_scope_is_relative_to_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert default_skills_root("project") == tmp_path / ".claude" / "skills"


class TestInstall:
    def test_installs_into_an_empty_directory(self, tmp_path):
        installed, message = install_skill(skills_root=tmp_path)
        assert installed is True
        assert (tmp_path / SKILL_NAME / "SKILL.md").is_file()
        assert (tmp_path / SKILL_NAME / "reference.md").is_file()
        assert "SKILL.md" in message

    def test_creates_missing_parent_directories(self, tmp_path):
        root = tmp_path / "does" / "not" / "exist"
        installed, _ = install_skill(skills_root=root)
        assert installed is True
        assert (root / SKILL_NAME / "SKILL.md").is_file()

    def test_reinstalling_an_unchanged_skill_is_a_no_op_success(self, tmp_path):
        install_skill(skills_root=tmp_path)
        installed, message = install_skill(skills_root=tmp_path)
        assert installed is True
        assert "Already up to date" in message

    def test_refuses_to_overwrite_an_edited_skill(self, tmp_path):
        install_skill(skills_root=tmp_path)
        edited = tmp_path / SKILL_NAME / "SKILL.md"
        edited.write_text("my own version", encoding="utf-8")

        installed, message = install_skill(skills_root=tmp_path)

        assert installed is False
        assert "SKILL.md" in message
        assert "--force" in message
        # The edit survives the refusal -- that is the entire point.
        assert edited.read_text(encoding="utf-8") == "my own version"

    def test_force_overwrites(self, tmp_path):
        install_skill(skills_root=tmp_path)
        edited = tmp_path / SKILL_NAME / "SKILL.md"
        edited.write_text("my own version", encoding="utf-8")

        installed, _ = install_skill(skills_root=tmp_path, force=True)

        assert installed is True
        assert edited.read_text(encoding="utf-8") != "my own version"

    def test_force_removes_files_the_package_no_longer_ships(self, tmp_path):
        """A stale file left behind would keep being read by the agent."""
        install_skill(skills_root=tmp_path)
        stale = tmp_path / SKILL_NAME / "old-notes.md"
        stale.write_text("from an older version", encoding="utf-8")

        install_skill(skills_root=tmp_path, force=True)

        assert not stale.exists()

    def test_an_extra_file_counts_as_a_difference(self, tmp_path):
        """Otherwise a directory with leftovers would report "up to date"."""
        install_skill(skills_root=tmp_path)
        (tmp_path / SKILL_NAME / "extra.md").write_text("x", encoding="utf-8")

        installed, message = install_skill(skills_root=tmp_path)

        assert installed is False
        assert "extra.md" in message

    def test_missing_package_skill_is_reported_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "zotero_mcp.skill_install.packaged_skill_dir",
            lambda: tmp_path / "nowhere",
        )
        installed, message = install_skill(skills_root=tmp_path)
        assert installed is False
        assert "missing" in message.lower()


class TestCliWiring:
    def test_install_skill_is_a_registered_command(self):
        """Reachable from the command line, not just importable."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "zotero_mcp.cli", "install-skill", "--help"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "--scope" in result.stdout
        assert "--force" in result.stdout

    def test_installing_via_the_cli_writes_the_files(self, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "zotero_mcp.cli", "install-skill",
             "--path", str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / SKILL_NAME / "SKILL.md").is_file()

    def test_a_refused_install_exits_non_zero(self, tmp_path):
        """A script installing the skill must be able to tell that it did not
        happen."""
        import subprocess
        import sys

        install_skill(skills_root=tmp_path)
        (tmp_path / SKILL_NAME / "SKILL.md").write_text("edited", encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "-m", "zotero_mcp.cli", "install-skill",
             "--path", str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 1
