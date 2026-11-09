"""`zotero-mcp setup` asks about WebDAV file sync and keeps it on a re-run, and the
write capabilities say where attachment files go."""

import json

import pytest
from conftest import DummyContext

from zotero_mcp import cli as _cli
from zotero_mcp import client as _client
from zotero_mcp import server, setup_helper

WEBDAV_KEYS = ("ZOTERO_WEBDAV_URL", "ZOTERO_WEBDAV_USERNAME", "ZOTERO_WEBDAV_PASSWORD")
SAVED = {
    "ZOTERO_WEBDAV_URL": "https://dav.example.com/zotero/",
    "ZOTERO_WEBDAV_USERNAME": "alice",
    "ZOTERO_WEBDAV_PASSWORD": "s3cret-pass",
}


class _Args:
    """Stand-in for the argparse namespace main() accepts: web mode, standalone config."""

    no_local = True
    no_claude = True
    api_key = "webkey"
    library_id = "123"
    library_type = "user"
    config_path = None
    skip_semantic_search = True
    semantic_config_only = False
    show_secrets = False


def _home(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(setup_helper, "find_executable", lambda: "/fake/bin/zotero-mcp")
    return tmp_path / ".config" / "zotero-mcp" / "config.json"


def _answers(monkeypatch, replies, password=None):
    """Feed input() from *replies*; past the end, input ends as it does for a script."""
    replies = list(replies)
    asked = []

    def _input(prompt=""):
        asked.append(prompt)
        if not replies:
            raise EOFError
        return replies.pop(0)

    def _getpass(prompt=""):
        asked.append(prompt)
        if password is None:
            raise EOFError
        return password

    monkeypatch.setattr("builtins.input", _input)
    monkeypatch.setattr(setup_helper.getpass, "getpass", _getpass)
    return asked


def _webdav_in(env):
    return {k: env[k] for k in WEBDAV_KEYS if k in env}


def _client_env(cfg_path):
    return json.loads(cfg_path.read_text(encoding="utf-8"))["client_env"]


class TestSetupAsksForWebDAV:
    def test_web_setup_stores_the_three_settings(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        _answers(monkeypatch, ["y", "dav.example.com", "alice"], password="s3cret-pass")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED
        out = capsys.readouterr().out
        assert "Files will go to: https://dav.example.com/zotero/" in out
        assert "WebDAV: https://dav.example.com/zotero/ (user alice)" in out
        assert "s3cret-pass" not in out

    def test_answering_no_stores_nothing(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        _answers(monkeypatch, ["n"])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {}
        assert "WebDAV: not used" in capsys.readouterr().out

    def test_end_of_input_means_no_webdav(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _answers(monkeypatch, [])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {}

    def test_settings_in_the_environment_are_used_without_asking(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        for key, value in SAVED.items():
            monkeypatch.setenv(key, value)
        asked = _answers(monkeypatch, [])

        assert setup_helper.main(_Args()) == 0

        assert asked == []
        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_claude_desktop_config_gets_the_settings(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        claude_cfg = tmp_path / "claude_desktop_config.json"
        args = _Args()
        args.no_claude = False
        args.config_path = str(claude_cfg)
        _answers(monkeypatch, ["y", "https://dav.example.com", "alice"], password="s3cret-pass")

        assert setup_helper.main(args) == 0

        env = json.loads(claude_cfg.read_text(encoding="utf-8"))["mcpServers"]["zotero"]["env"]
        assert _webdav_in(env) == SAVED

    def test_group_library_is_not_asked(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        args = _Args()
        args.library_type = "group"
        asked = _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert asked == []
        assert _webdav_in(_client_env(cfg)) == {}

    def test_local_mode_is_not_asked(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        args = _Args()
        args.no_local = False
        asked = _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert asked == []
        assert _webdav_in(_client_env(cfg)) == {}


class TestSetupKeepsWebDAVOnRerun:
    """setup rewrites the config's env section from scratch; settings a user put
    there must survive a re-run unless the user drops them."""

    @staticmethod
    def _saved_config(cfg):
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"client_env": {"ZOTERO_LOCAL": "false", **SAVED}}), encoding="utf-8")

    def test_enter_keeps_them(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        self._saved_config(cfg)
        _answers(monkeypatch, [""])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_end_of_input_keeps_them(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        self._saved_config(cfg)
        _answers(monkeypatch, [])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_declining_both_questions_removes_them(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        self._saved_config(cfg)
        _answers(monkeypatch, ["n", "n"])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {}

    def test_claude_desktop_rerun_keeps_them(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        claude_cfg = tmp_path / "claude_desktop_config.json"
        entry = {"command": "/fake/bin/zotero-mcp", "env": {"ZOTERO_LOCAL": "false", **SAVED}}
        claude_cfg.write_text(json.dumps({"mcpServers": {"zotero": entry}}), encoding="utf-8")
        args = _Args()
        args.no_claude = False
        args.config_path = str(claude_cfg)
        _answers(monkeypatch, [""])

        assert setup_helper.main(args) == 0

        env = json.loads(claude_cfg.read_text(encoding="utf-8"))["mcpServers"]["zotero"]["env"]
        assert _webdav_in(env) == SAVED

    def test_local_mode_rerun_keeps_them(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        self._saved_config(cfg)
        args = _Args()
        args.no_local = False
        _answers(monkeypatch, [""])

        assert setup_helper.main(args) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED


@pytest.mark.parametrize(
    "address, folder",
    [
        ("dav.example.com", "https://dav.example.com/zotero/"),
        ("https://dav.example.com", "https://dav.example.com/zotero/"),
        ("https://dav.example.com/", "https://dav.example.com/zotero/"),
        ("http://nas.local:5005", "http://nas.local:5005/zotero/"),
        ("nas.local:5006/zotero", "https://nas.local:5006/zotero/zotero/"),  # a share named zotero
        (
            "https://cloud.example.com/remote.php/dav/files/alice",
            "https://cloud.example.com/remote.php/dav/files/alice/zotero/",
        ),
    ],
)
def test_the_address_from_zoteros_settings_becomes_its_zotero_folder(address, folder):
    assert setup_helper._zotero_folder_url(address) == folder


class TestFileRoute:
    @pytest.fixture
    def web(self, monkeypatch):
        monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
        monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
        monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")

    @pytest.fixture
    def webdav(self, monkeypatch):
        for key, value in SAVED.items():
            monkeypatch.setenv(key, value)

    def test_webdav(self, web, webdav):
        assert _client.get_write_capabilities(probe=False)["file_route"] == "webdav"

    def test_zotero_storage_without_webdav(self, web):
        assert _client.get_write_capabilities(probe=False)["file_route"] == "zotero_storage"

    def test_a_group_library_uses_zotero_storage_even_with_webdav(self, web, webdav, monkeypatch):
        monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")
        assert _client.get_write_capabilities(probe=False)["file_route"] == "zotero_storage_group"

    def test_local_writes_hand_files_to_the_running_zotero(self, monkeypatch, webdav):
        monkeypatch.setenv("ZOTERO_LOCAL", "true")
        monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", "k")
        caps = _client.get_write_capabilities(probe=False)
        assert caps["mode"] == "local"
        assert caps["file_route"] == "local"

    def test_no_route_when_writes_are_impossible(self, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL", "true")
        monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
        monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
        caps = _client.get_write_capabilities(probe=False)
        assert caps["mode"] == "none"
        assert caps["file_route"] is None

    def test_setup_info_prints_the_route(self, web, webdav, capsys):
        _cli._print_write_capabilities(_client.get_write_capabilities(probe=False))
        out = capsys.readouterr().out
        assert "File storage:          WebDAV (uploads go straight there" in out
        assert "dav.example.com" not in out  # setup-info masks the WebDAV URL

    def test_the_mcp_tool_reports_the_route(self, web, webdav):
        result = server.write_capabilities(ctx=DummyContext())
        assert "File storage: WebDAV (uploads go straight there" in result
        assert "dav.example.com" not in result
