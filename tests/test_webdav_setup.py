"""`zotero-mcp setup` keeps the WebDAV settings in the configs it rewrites."""

import json

import pytest

from zotero_mcp import setup_helper

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


def _answers(monkeypatch, replies, password=None, end=EOFError):
    """Feed input() from *replies* and getpass() *password*.

    Past the end of *replies*, and for a password of None, *end* is raised:
    EOFError as for a script, KeyboardInterrupt as for Ctrl-C. Returns the
    prompts shown, in order.
    """
    replies = list(replies)
    asked = []

    def _input(prompt=""):
        asked.append(prompt)
        if not replies:
            raise end
        return replies.pop(0)

    def _getpass(prompt=""):
        asked.append(prompt)
        if password is None:
            raise end
        return password

    monkeypatch.setattr("builtins.input", _input)
    monkeypatch.setattr(setup_helper.getpass, "getpass", _getpass)
    return asked


def _webdav_in(env):
    return {k: env[k] for k in WEBDAV_KEYS if k in env}


def _client_env(cfg_path):
    return json.loads(cfg_path.read_text(encoding="utf-8"))["client_env"]


def _claude_env(claude_cfg):
    return json.loads(claude_cfg.read_text(encoding="utf-8"))["mcpServers"]["zotero"]["env"]


def _write_claude(claude_cfg, env):
    entry = {"command": "/fake/bin/zotero-mcp", "env": env}
    claude_cfg.parent.mkdir(parents=True, exist_ok=True)
    claude_cfg.write_text(json.dumps({"mcpServers": {"zotero": entry}}), encoding="utf-8")


def _saved_config(cfg, webdav=SAVED):
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"client_env": {"ZOTERO_LOCAL": "false", **webdav}}), encoding="utf-8")


class TestSetupKeepsWebDAVOnRerun:
    """setup rebuilds the env section of the configs it writes; WebDAV settings
    in them must survive a re-run."""

    @pytest.mark.parametrize("change", [{}, {"no_local": False}, {"library_type": "group"}])
    def test_a_rerun_keeps_them_in_every_mode(self, monkeypatch, tmp_path, change):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        args = _Args()
        for name, value in change.items():
            setattr(args, name, value)
        asked = _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert asked == []
        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_claude_desktop_rerun_keeps_them(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        claude_cfg = tmp_path / "claude_desktop_config.json"
        _write_claude(claude_cfg, {"ZOTERO_LOCAL": "false", **SAVED})
        args = _Args()
        args.no_claude = False
        args.config_path = str(claude_cfg)
        _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert _webdav_in(_claude_env(claude_cfg)) == SAVED

    def test_a_run_that_does_not_ask_keeps_a_partial_set_as_it_is(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        partial = {"ZOTERO_WEBDAV_URL": SAVED["ZOTERO_WEBDAV_URL"]}
        _saved_config(cfg, partial)
        args = _Args()
        args.no_local = False
        _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert _webdav_in(_client_env(cfg)) == partial

    @staticmethod
    def _two_claude_configs(monkeypatch, tmp_path, first_env, second_env):
        first = tmp_path / "Claude" / "claude_desktop_config.json"
        second = tmp_path / "Claude-3p" / "claude_desktop_config.json"
        _write_claude(first, first_env)
        _write_claude(second, second_env)
        monkeypatch.setattr(setup_helper, "find_all_claude_configs", lambda verbose=False: [first, second])
        args = _Args()
        args.no_claude = False
        return first, second, args

    def test_each_claude_config_keeps_its_own_when_setup_does_not_ask(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        first, second, args = self._two_claude_configs(
            monkeypatch, tmp_path, {"ZOTERO_LOCAL": "true", **SAVED}, {"ZOTERO_LOCAL": "true"}
        )
        args.no_local = False
        _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert _webdav_in(_claude_env(first)) == SAVED
        assert _webdav_in(_claude_env(second)) == {}

    @pytest.mark.parametrize("client_env", [None, "oops", ["ZOTERO_WEBDAV_URL"]])
    def test_an_env_section_of_the_wrong_shape_is_rebuilt_without_webdav(self, monkeypatch, tmp_path, client_env):
        cfg = _home(monkeypatch, tmp_path)
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"client_env": client_env}), encoding="utf-8")
        args = _Args()
        args.no_local = False
        _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert _webdav_in(_client_env(cfg)) == {}
