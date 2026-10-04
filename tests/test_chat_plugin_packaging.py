"""`zotero-cli plugin`, and the files that get the Zotero chat plugin to users.

The xpi is built from plugin/ at release time and is not committed, so what
can break unnoticed is the plumbing: the command that finds it, the ignore and
hatch rules that carry it into the wheel, and the update manifest.
"""

import argparse
import json
from pathlib import Path

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp.cli_standalone import _CMD_MAP, build_parser

REPO = Path(__file__).resolve().parent.parent
XPI = "src/zotero_mcp/chat_plugin/zotero-chat.xpi"


def _run(monkeypatch, root: Path, *, path=False, json_out=False, reveal=False):
    """Run the command as if the package lived at <root>/src/zotero_mcp."""
    monkeypatch.setattr(cli_standalone, "__file__", str(root / "src" / "zotero_mcp" / "cli_standalone.py"))
    cli_standalone.cmd_plugin(argparse.Namespace(path=path, reveal=reveal, json_out=json_out, verbose=False))


def _xpi_in(root: Path, rel: str) -> Path:
    xpi = root / rel
    xpi.parent.mkdir(parents=True)
    xpi.write_bytes(b"PK")
    return xpi


class TestParsing:
    def test_routes_and_takes_global_flags_either_side(self):
        parser = build_parser()
        assert "plugin" in _CMD_MAP
        assert parser.parse_args(["plugin"]).path is False
        assert parser.parse_args(["plugin", "--path"]).path is True
        assert parser.parse_args(["plugin"]).reveal is False
        assert parser.parse_args(["plugin", "--reveal"]).reveal is True
        assert parser.parse_args(["--json", "plugin"]).json_out is True
        assert parser.parse_args(["plugin", "--json"]).json_out is True


class TestPluginCommand:
    def test_prints_path_and_install_steps(self, tmp_path, monkeypatch, capsys):
        xpi = _xpi_in(tmp_path, XPI)
        _run(monkeypatch, tmp_path)
        out = capsys.readouterr().out
        assert str(xpi) in out
        assert "Install Plugin From File" in out

    def test_path_flag_prints_only_the_path(self, tmp_path, monkeypatch, capsys):
        xpi = _xpi_in(tmp_path, XPI)
        _run(monkeypatch, tmp_path, path=True)
        assert capsys.readouterr().out == f"{xpi}\n"

    def test_reveal_shows_the_file_and_still_prints_the_steps(self, tmp_path, monkeypatch, capsys):
        xpi = _xpi_in(tmp_path, XPI)
        shown = []
        monkeypatch.setattr(cli_standalone, "_reveal", shown.append)
        _run(monkeypatch, tmp_path, reveal=True)
        assert shown == [xpi]
        assert "Install Plugin From File" in capsys.readouterr().out

    def test_source_checkout_build_is_found(self, tmp_path, monkeypatch, capsys):
        xpi = _xpi_in(tmp_path, "plugin/dist/zotero-chat.xpi")
        _run(monkeypatch, tmp_path, path=True)
        assert capsys.readouterr().out == f"{xpi}\n"

    def test_packaged_copy_wins_over_a_checkout_build(self, tmp_path, monkeypatch, capsys):
        _xpi_in(tmp_path, "plugin/dist/zotero-chat.xpi")
        packaged = _xpi_in(tmp_path, XPI)
        _run(monkeypatch, tmp_path, path=True)
        assert capsys.readouterr().out == f"{packaged}\n"

    @pytest.mark.parametrize("path", [False, True])
    def test_missing_xpi_exits_nonzero_with_build_hint(self, tmp_path, monkeypatch, capsys, path):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, tmp_path, path=path)
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""  # nothing a script could mistake for a path
        assert "npm run build" in captured.err
        assert "releases/latest" in captured.err

    def test_json_success(self, tmp_path, monkeypatch, capsys):
        xpi = _xpi_in(tmp_path, XPI)
        _run(monkeypatch, tmp_path, json_out=True)
        env = json.loads(capsys.readouterr().out)
        assert env["ok"] is True and env["command"] == "plugin"
        assert env["data"] == {"path": str(xpi)}

    def test_json_missing(self, tmp_path, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, tmp_path, json_out=True)
        assert exc.value.code == 1
        env = json.loads(capsys.readouterr().out)
        assert env["ok"] is False
        assert env["error"]["code"] == "plugin_missing"


class TestShipping:
    def test_xpi_is_ignored_by_git_but_forced_into_the_build(self):
        """Gitignored files are left out of hatch builds unless named in
        `artifacts`; at the top level, because the wheel is built from the sdist."""
        assert XPI in (REPO / ".gitignore").read_text().splitlines()
        pyproject = (REPO / "pyproject.toml").read_text()
        assert f'[tool.hatch.build]\nartifacts = ["{XPI}"]' in pyproject

    def test_cli_default_name_matches_the_built_one(self):
        """build.mjs names the file <slug>.xpi."""
        slug = json.loads((REPO / "plugin" / "addon.json").read_text())["slug"]
        assert XPI.endswith(f"/{slug}.xpi")


class TestUpdateManifest:
    addon = json.loads((REPO / "plugin" / "addon.json").read_text())
    updates = json.loads((REPO / "plugin" / "updates.json").read_text())

    def test_addon_points_at_the_committed_manifest(self):
        assert self.addon["updateUrl"].endswith("/main/plugin/updates.json")
        assert self.addon["updateUrl"].startswith("https://raw.githubusercontent.com/54yyyu/zotero-mcp/")

    def test_manifest_is_for_the_addon_id(self):
        assert list(self.updates["addons"]) == [self.addon["id"]]

    def test_each_update_is_well_formed(self):
        for u in self.updates["addons"][self.addon["id"]]["updates"]:
            assert u["version"]
            assert u["update_link"].startswith("https://")
            assert u["update_link"].endswith(f"/{self.addon['slug']}.xpi")
            assert u["update_hash"].startswith("sha256:")
            assert u["applications"]["zotero"]["strict_min_version"] == self.addon["minZotero"]
