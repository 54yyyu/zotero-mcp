"""`zotero-mcp setup` asks about WebDAV file sync and keeps the settings on a re-run."""

import json

import pytest

from zotero_mcp import setup_helper

WEBDAV_KEYS = ("ZOTERO_WEBDAV_URL", "ZOTERO_WEBDAV_USERNAME", "ZOTERO_WEBDAV_PASSWORD")
SAVED = {
    "ZOTERO_WEBDAV_URL": "https://dav.example.com/zotero/",
    "ZOTERO_WEBDAV_USERNAME": "alice",
    "ZOTERO_WEBDAV_PASSWORD": "s3cret-pass",
}
FRESH = ["y", "dav.example.com", "alice"]  # then the password


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


class TestSetupAsksForWebDAV:
    def test_web_setup_stores_the_three_settings(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        _answers(monkeypatch, FRESH, password="s3cret-pass")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED
        out = capsys.readouterr().out
        assert "Files will go to: https://dav.example.com/zotero/" in out
        assert "WebDAV file sync: yes" in out
        assert "s3cret-pass" not in out

    def test_the_printed_client_env_masks_all_three(self, monkeypatch, tmp_path, capsys):
        _home(monkeypatch, tmp_path)
        _answers(monkeypatch, FRESH, password="s3cret-pass")

        assert setup_helper.main(_Args()) == 0

        out = capsys.readouterr().out
        shown = json.loads(next(line for line in out.splitlines() if line.startswith('{"ZOTERO_LOCAL"')))
        for key, value in SAVED.items():
            assert shown[key] != value

    @pytest.mark.parametrize("yes", ["y", "yes", "Y", " YES "])
    def test_yes_in_any_case_is_yes(self, monkeypatch, tmp_path, yes):
        cfg = _home(monkeypatch, tmp_path)
        _answers(monkeypatch, [yes, *FRESH[1:]], password="s3cret-pass")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED

    @pytest.mark.parametrize("no", ["n", "no", "N"])
    def test_answering_no_stores_nothing(self, monkeypatch, tmp_path, capsys, no):
        cfg = _home(monkeypatch, tmp_path)
        asked = _answers(monkeypatch, [no])

        assert setup_helper.main(_Args()) == 0

        assert len(asked) == 1
        assert _webdav_in(_client_env(cfg)) == {}
        assert "WebDAV file sync: no" in capsys.readouterr().out

    def test_enter_passes_the_question_when_nothing_is_stored(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        asked = _answers(monkeypatch, [""])

        assert setup_helper.main(_Args()) == 0

        assert len(asked) == 1
        assert "[y/N]" in asked[0]
        assert _webdav_in(_client_env(cfg)) == {}
        assert "WebDAV file sync" not in capsys.readouterr().out

    def test_an_answer_that_is_neither_is_asked_again(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        asked = _answers(monkeypatch, ["ye", "1", *FRESH], password="s3cret-pass")

        assert setup_helper.main(_Args()) == 0

        assert asked[0] == asked[1] == asked[2]
        assert capsys.readouterr().out.count("Please enter y or n") == 2
        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_end_of_input_stores_nothing_and_setup_completes(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _answers(monkeypatch, [])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {}

    @pytest.mark.parametrize(
        "replies, password, prompts",
        [
            (["y", ""], "s3cret-pass", 2),  # no address: the step ends there
            (["y", "dav.example.com", ""], "s3cret-pass", 4),
            (["y", "dav.example.com", "alice"], "", 4),
        ],
    )
    def test_a_blank_answer_stores_nothing(self, monkeypatch, tmp_path, capsys, replies, password, prompts):
        cfg = _home(monkeypatch, tmp_path)
        asked = _answers(monkeypatch, replies, password=password)

        assert setup_helper.main(_Args()) == 0

        assert len(asked) == prompts
        assert _webdav_in(_client_env(cfg)) == {}
        out = capsys.readouterr().out
        assert "WebDAV settings not changed" in out
        assert "WebDAV file sync" not in out

    @pytest.mark.parametrize(
        "address, problem",
        [
            ("davs://dav.example.com", "it is not an http or https address"),
            ("webdav://dav.example.com", "it is not an http or https address"),
            ("https:///dav", "it is not an http or https address"),
            ("https://[::1", "it is not an http or https address"),
            (
                "https://alice:hunter2@dav.example.com",
                "the username and the password are asked for next, so leave them out of it",
            ),
        ],
    )
    def test_an_address_that_cannot_work_is_asked_again(
        self, monkeypatch, tmp_path, capsys, address, problem
    ):
        cfg = _home(monkeypatch, tmp_path)
        asked = _answers(monkeypatch, ["y", address, *FRESH[1:]], password="s3cret-pass")

        assert setup_helper.main(_Args()) == 0

        assert asked[1] == asked[2]  # the address, twice
        assert len(asked) == 5  # and the password only once
        assert _webdav_in(_client_env(cfg)) == SAVED
        out = capsys.readouterr().out
        assert f"That address cannot be used: {problem}." in out
        assert "hunter2" not in out

    @pytest.mark.parametrize(
        "typed", ["https://dav.example.com/zotero/", "https://dav.example.com/zotero", "dav.example.com/"]
    )
    def test_the_folder_typed_in_full_is_not_doubled(self, monkeypatch, tmp_path, typed):
        cfg = _home(monkeypatch, tmp_path)
        _answers(monkeypatch, ["y", typed, "alice"], password="s3cret-pass")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_settings_in_the_environment_are_used_without_asking(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        for key, value in SAVED.items():
            monkeypatch.setenv(key, value)
        asked = _answers(monkeypatch, [])

        assert setup_helper.main(_Args()) == 0

        assert asked == []
        assert _webdav_in(_client_env(cfg)) == SAVED
        assert "s3cret-pass" not in capsys.readouterr().out

    @pytest.mark.parametrize(
        "env",
        [
            {"ZOTERO_WEBDAV_URL": SAVED["ZOTERO_WEBDAV_URL"]},
            {**SAVED, "ZOTERO_WEBDAV_PASSWORD": "  "},
            {**SAVED, "ZOTERO_WEBDAV_PASSWORD": "REPLACE_WITH_YOUR_PASSWORD"},
        ],
    )
    def test_an_incomplete_set_in_the_environment_is_not_used(self, monkeypatch, tmp_path, env):
        cfg = _home(monkeypatch, tmp_path)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        asked = _answers(monkeypatch, [""])

        assert setup_helper.main(_Args()) == 0

        assert "[y/N]" in asked[0]
        assert _webdav_in(_client_env(cfg)) == {}

    def test_claude_desktop_config_gets_the_settings(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        claude_cfg = tmp_path / "claude_desktop_config.json"
        args = _Args()
        args.no_claude = False
        args.config_path = str(claude_cfg)
        _answers(monkeypatch, ["y", "https://dav.example.com", "alice"], password="s3cret-pass")

        assert setup_helper.main(args) == 0

        assert _webdav_in(_claude_env(claude_cfg)) == SAVED

    @pytest.mark.parametrize("change", [{"no_local": False}, {"library_type": "group"}])
    def test_local_mode_and_a_group_library_are_not_asked(self, monkeypatch, tmp_path, capsys, change):
        cfg = _home(monkeypatch, tmp_path)
        args = _Args()
        for name, value in change.items():
            setattr(args, name, value)
        asked = _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert asked == []
        assert _webdav_in(_client_env(cfg)) == {}
        assert "WebDAV" not in capsys.readouterr().out

    @pytest.mark.parametrize("stored", [None, SAVED])
    def test_a_script_that_answers_only_the_earlier_questions_still_works(self, monkeypatch, tmp_path, stored):
        """The WebDAV question is the last one, so the script reaches end of input there."""
        cfg = _home(monkeypatch, tmp_path)
        if stored:
            _saved_config(cfg, stored)
        args = _Args()
        args.skip_semantic_search = False
        asked = _answers(monkeypatch, ["n"])  # "configure semantic search?"

        assert setup_helper.main(args) == 0

        assert len(asked) == 2
        assert "Do you sync attachment files over WebDAV?" in asked[1]
        assert _webdav_in(_client_env(cfg)) == (stored or {})

    @pytest.mark.parametrize("existing", [None, SAVED])
    def test_the_step_prints_ascii_only(self, monkeypatch, capsys, existing):
        """A redirected Windows console cannot encode an arrow or a dash."""
        asked = _answers(monkeypatch, ["maybe", "y", "davs://x", *FRESH[1:]], password="s3cret-pass")

        setup_helper.setup_webdav(existing)

        assert capsys.readouterr().out.isascii()
        assert all(prompt.isascii() for prompt in asked)


class TestSetupKeepsWebDAVOnRerun:
    """setup rebuilds the env section of the configs it writes; WebDAV settings
    in them must survive a re-run unless the user drops them."""

    def test_enter_at_every_prompt_keeps_them(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        asked = _answers(monkeypatch, ["", "", ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED
        assert "[Y/n]" in asked[0]
        assert "WebDAV file sync: yes" in capsys.readouterr().out

    def test_the_stored_values_are_the_defaults(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        asked = _answers(monkeypatch, ["", "", ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert "[https://dav.example.com/zotero/]" in asked[1]
        assert "[alice]" in asked[2]
        assert "Enter keeps the stored password" in asked[3]
        assert all("s3cret-pass" not in prompt for prompt in asked)

    def test_a_new_password_alone_is_stored(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        _answers(monkeypatch, ["", "", ""], password="new-pass")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {**SAVED, "ZOTERO_WEBDAV_PASSWORD": "new-pass"}

    def test_a_new_address_gets_its_zotero_folder(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        _answers(monkeypatch, ["y", "https://other.example.org", ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {
            **SAVED,
            "ZOTERO_WEBDAV_URL": "https://other.example.org/zotero/",
        }

    @pytest.mark.parametrize(
        "folder", ["https://dav.example.com/files/", "https://dav.example.com/zotero/zotero/"]
    )
    def test_enter_keeps_a_stored_folder_whatever_it_is_called(self, monkeypatch, tmp_path, folder):
        cfg = _home(monkeypatch, tmp_path)
        stored = {**SAVED, "ZOTERO_WEBDAV_URL": folder}
        _saved_config(cfg, stored)
        asked = _answers(monkeypatch, ["", "", ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert f"[{folder}]" in asked[1]
        assert _webdav_in(_client_env(cfg)) == stored

    @pytest.mark.parametrize(
        "folder, noted",
        [
            ("https://dav.example.com/", True),
            ("https://dav.example.com/files/", True),
            ("https://dav.example.com/zotero/", False),
            ("https://dav.example.com/zotero", False),
        ],
    )
    def test_a_stored_folder_zotero_cannot_read_is_pointed_out(
        self, monkeypatch, tmp_path, capsys, folder, noted
    ):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg, {**SAVED, "ZOTERO_WEBDAV_URL": folder})
        _answers(monkeypatch, ["", "", ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert ("does not end in /zotero/" in capsys.readouterr().out) is noted

    def test_the_shown_folder_typed_again_is_the_same_folder(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        _answers(monkeypatch, ["", SAVED["ZOTERO_WEBDAV_URL"], ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED

    @pytest.mark.parametrize("end", [EOFError, KeyboardInterrupt])
    @pytest.mark.parametrize("replies", [[], ["y"], ["y", ""], ["y", "", ""]])
    def test_end_of_input_or_ctrl_c_at_any_prompt_keeps_them(self, monkeypatch, tmp_path, replies, end):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        _answers(monkeypatch, replies, end=end)

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_a_mistyped_answer_does_not_remove_them(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        asked = _answers(monkeypatch, ["ye", "", "", ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert asked[0] == asked[1]
        assert _webdav_in(_client_env(cfg)) == SAVED

    def test_an_address_that_cannot_work_does_not_remove_them(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        _answers(monkeypatch, ["y", "davs://dav.example.com", "", ""], password="")

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == SAVED

    @pytest.mark.parametrize("no", ["n", "no"])
    def test_answering_no_removes_them(self, monkeypatch, tmp_path, capsys, no):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        _answers(monkeypatch, [no])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {}
        assert "WebDAV file sync: no" in capsys.readouterr().out

    def test_claude_desktop_rerun_keeps_them(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        claude_cfg = tmp_path / "claude_desktop_config.json"
        _write_claude(claude_cfg, {"ZOTERO_LOCAL": "false", **SAVED})
        args = _Args()
        args.no_claude = False
        args.config_path = str(claude_cfg)
        _answers(monkeypatch, ["", "", ""], password="")

        assert setup_helper.main(args) == 0

        assert _webdav_in(_claude_env(claude_cfg)) == SAVED

    @pytest.mark.parametrize("change", [{"no_local": False}, {"library_type": "group"}])
    def test_a_run_that_does_not_ask_keeps_them(self, monkeypatch, tmp_path, capsys, change):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg)
        args = _Args()
        for name, value in change.items():
            setattr(args, name, value)
        asked = _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert asked == []
        assert _webdav_in(_client_env(cfg)) == SAVED
        assert "WebDAV file sync" not in capsys.readouterr().out

    def test_a_run_that_does_not_ask_keeps_a_partial_set_as_it_is(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        partial = {"ZOTERO_WEBDAV_URL": SAVED["ZOTERO_WEBDAV_URL"]}
        _saved_config(cfg, partial)
        args = _Args()
        args.no_local = False
        _answers(monkeypatch, [])

        assert setup_helper.main(args) == 0

        assert _webdav_in(_client_env(cfg)) == partial

    @pytest.mark.parametrize(
        "stored",
        [
            {"ZOTERO_WEBDAV_URL": SAVED["ZOTERO_WEBDAV_URL"], "ZOTERO_WEBDAV_USERNAME": "alice"},
            {**SAVED, "ZOTERO_WEBDAV_PASSWORD": "REPLACE_WITH_YOUR_PASSWORD"},
        ],
    )
    def test_an_incomplete_stored_set_is_not_offered_and_enter_leaves_it(self, monkeypatch, tmp_path, stored):
        """The rest may be in the shell that runs the server."""
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg, stored)
        asked = _answers(monkeypatch, [""])

        assert setup_helper.main(_Args()) == 0

        assert "[y/N]" in asked[0]
        assert _webdav_in(_client_env(cfg)) == stored

    @pytest.mark.parametrize(
        "replies, password",
        [(["y", ""], None), (["y", "dav.example.com", ""], "s3cret-pass"), (["y", "dav.example.com", "alice"], "")],
    )
    def test_a_blank_answer_leaves_an_incomplete_stored_set(self, monkeypatch, tmp_path, replies, password):
        cfg = _home(monkeypatch, tmp_path)
        stored = {"ZOTERO_WEBDAV_URL": SAVED["ZOTERO_WEBDAV_URL"]}
        _saved_config(cfg, stored)
        _answers(monkeypatch, replies, password=password)

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == stored

    def test_a_stored_address_with_a_login_in_it_is_never_shown(self, monkeypatch, tmp_path, capsys):
        cfg = _home(monkeypatch, tmp_path)
        stored = {**SAVED, "ZOTERO_WEBDAV_URL": "https://alice:hunter2@dav.example.com/zotero/"}
        _saved_config(cfg, stored)
        asked = _answers(monkeypatch, [""])

        assert setup_helper.main(_Args()) == 0

        assert "[y/N]" in asked[0]
        assert "hunter2" not in "".join(asked) + capsys.readouterr().out
        assert _webdav_in(_client_env(cfg)) == stored

    def test_answering_no_removes_an_incomplete_stored_set(self, monkeypatch, tmp_path):
        cfg = _home(monkeypatch, tmp_path)
        _saved_config(cfg, {"ZOTERO_WEBDAV_URL": SAVED["ZOTERO_WEBDAV_URL"]})
        _answers(monkeypatch, ["n"])

        assert setup_helper.main(_Args()) == 0

        assert _webdav_in(_client_env(cfg)) == {}

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

    def test_the_defaults_come_from_whichever_claude_config_has_them(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        first, second, args = self._two_claude_configs(
            monkeypatch, tmp_path, {"ZOTERO_LOCAL": "false"}, {"ZOTERO_LOCAL": "false", **SAVED}
        )
        asked = _answers(monkeypatch, ["", "", ""], password="")

        assert setup_helper.main(args) == 0

        assert "[Y/n]" in asked[0]
        assert _webdav_in(_claude_env(first)) == SAVED
        assert _webdav_in(_claude_env(second)) == SAVED

    def test_a_claude_config_that_cannot_be_read_does_not_hide_the_next_one(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        first, second, args = self._two_claude_configs(
            monkeypatch, tmp_path, {}, {"ZOTERO_LOCAL": "false", **SAVED}
        )
        first.write_text("not json", encoding="utf-8")
        asked = _answers(monkeypatch, ["", "", ""], password="")

        assert setup_helper.main(args) == 0

        assert "[Y/n]" in asked[0]
        assert _webdav_in(_claude_env(second)) == SAVED

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

    def test_answering_no_removes_them_from_every_claude_config(self, monkeypatch, tmp_path):
        _home(monkeypatch, tmp_path)
        first, second, args = self._two_claude_configs(
            monkeypatch, tmp_path, {"ZOTERO_LOCAL": "false", **SAVED}, {"ZOTERO_LOCAL": "false", **SAVED}
        )
        _answers(monkeypatch, ["n"])

        assert setup_helper.main(args) == 0

        assert _webdav_in(_claude_env(first)) == {}
        assert _webdav_in(_claude_env(second)) == {}


@pytest.mark.parametrize(
    "address, folder",
    [
        ("dav.example.com", "https://dav.example.com/zotero/"),
        ("  dav.example.com  ", "https://dav.example.com/zotero/"),
        ("https://dav.example.com", "https://dav.example.com/zotero/"),
        ("https://dav.example.com/", "https://dav.example.com/zotero/"),
        ("http://nas.local:5005", "http://nas.local:5005/zotero/"),
        # Zotero drops a trailing /zotero from what is typed, so a share of that name is typed twice
        ("https://dav.example.com/zotero", "https://dav.example.com/zotero/"),
        ("https://dav.example.com/zotero/", "https://dav.example.com/zotero/"),
        ("https://dav.example.com//", "https://dav.example.com/zotero/"),
        ("https://dav.example.com/zotero//", "https://dav.example.com/zotero/"),
        # A host called zotero is not a /zotero folder
        ("https://zotero", "https://zotero/zotero/"),
        ("zotero:8080", "https://zotero:8080/zotero/"),
        ("nas.local:5006/zotero/zotero", "https://nas.local:5006/zotero/zotero/"),
        ("https://dav.example.com/Zotero", "https://dav.example.com/Zotero/zotero/"),
        (
            "https://cloud.example.com/remote.php/dav/files/alice",
            "https://cloud.example.com/remote.php/dav/files/alice/zotero/",
        ),
    ],
)
def test_the_address_from_zoteros_settings_becomes_its_zotero_folder(address, folder):
    assert setup_helper._zotero_folder_url(address) == folder
