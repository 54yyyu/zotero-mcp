"""Regression tests: `zotero-mcp update` must report what is installed, not what PyPI has.

`update_via_method("uv")` ran `uv tool upgrade` and took exit 0 as "updated".
But a tool installed with an exact pin (`uv tool install "zotero-mcp-server[all]==0.9.0"`)
records that pin in its receipt, and `uv tool upgrade` then does nothing, prints
"Nothing to upgrade", and still exits 0. The updater went on to announce
"Successfully updated from 0.9.0 to 0.11.0" — a message built from the PyPI
lookup, with nothing re-read from disk — while 0.9.0 stayed installed.

Two rules follow. The outcome must come from the version actually installed
after the update, read in a fresh interpreter. And a uv tool install whose
receipt carries a version specifier must be reinstalled with `@latest`,
keeping its extras and Python, because `uv tool upgrade` cannot move it.
"""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from zotero_mcp import updater


@pytest.fixture
def update_scaffold(monkeypatch, tmp_path):
    """Stub everything around the install step: versions, backup, restore, verify."""
    backup = tmp_path / "backup"
    backup.mkdir()
    monkeypatch.setattr(updater, "get_current_version", lambda: "0.9.0")
    monkeypatch.setattr(updater, "get_latest_version", lambda: "0.11.0")
    monkeypatch.setattr(updater, "detect_installation_method", lambda: "uv")
    monkeypatch.setattr(updater, "backup_configurations", lambda: backup)
    monkeypatch.setattr(updater, "restore_configurations", lambda _backup: True)
    monkeypatch.setattr(updater, "verify_installation", lambda python=None: (True, "verified"))
    monkeypatch.setattr(updater, "_probe_python", lambda _method: None)
    monkeypatch.setattr(
        updater, "update_via_method", lambda _method, force=False: (True, "Updated successfully via uv tool")
    )


class TestUpdateReportsInstalledVersion:
    def test_unchanged_install_is_a_failure(self, update_scaffold, monkeypatch):
        """The user-visible bug: the install step 'succeeded' but nothing changed."""
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.9.0")

        result = updater.update_zotero_mcp()

        assert result["success"] is False
        assert result["installed_version"] == "0.9.0"
        assert "0.9.0" in result["message"]
        assert "Successfully" not in result["message"]

    def test_success_names_the_installed_version(self, update_scaffold, monkeypatch):
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.11.0")

        result = updater.update_zotero_mcp()

        assert result["success"] is True
        assert result["installed_version"] == "0.11.0"
        assert "0.9.0" in result["message"] and "0.11.0" in result["message"]

    def test_unreadable_installed_version_is_a_failure(self, update_scaffold, monkeypatch):
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: None)

        result = updater.update_zotero_mcp()

        assert result["success"] is False


class TestUvToolReceipt:
    def test_reinstall_command_preserves_extras_and_python(self):
        receipt = {"extras": ["all"], "specifier": "==0.9.0", "python": "3.12"}

        assert updater._uv_tool_reinstall_command(receipt) == [
            "uv", "tool", "install", "--force", "--python", "3.12", "zotero-mcp-server[all]@latest",
        ]

    def test_reinstall_command_bare(self):
        receipt = {"extras": [], "specifier": "", "python": None}

        assert updater._uv_tool_reinstall_command(receipt) == [
            "uv", "tool", "install", "--force", "zotero-mcp-server@latest",
        ]

    def test_read_receipt(self, tmp_path, monkeypatch):
        """Parse the receipt uv writes; this is the 0.9.0 pin verbatim."""
        tool_dir = tmp_path / "zotero-mcp-server"
        tool_dir.mkdir()
        (tool_dir / "uv-receipt.toml").write_text(
            '[tool]\n'
            'requirements = [{ name = "zotero-mcp-server", extras = ["all"], specifier = "==0.9.0" }]\n'
            'python = "3.12"\n'
            'entrypoints = [\n'
            '    { name = "zotero-mcp", install-path = "/x/bin/zotero-mcp", from = "zotero-mcp-server" },\n'
            ']\n'
        )
        monkeypatch.setattr(updater, "_uv_tool_dir", lambda: tmp_path)

        assert updater._read_uv_tool_receipt() == {
            "extras": ["all"], "specifier": "==0.9.0", "python": "3.12",
        }

    def test_read_receipt_missing_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(updater, "_uv_tool_dir", lambda: tmp_path)

        assert updater._read_uv_tool_receipt() is None


class TestUvToolUpdatePath:
    @pytest.fixture
    def runner(self, monkeypatch):
        """Record every command the uv path runs; all of them 'succeed'."""
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="Nothing to upgrade\n", stderr="")

        monkeypatch.setattr(updater.subprocess, "run", fake_run)
        monkeypatch.setattr(updater, "_is_uv_tool_installation", lambda: True)
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(updater, "_uv_tool_python", lambda _d=None: None)
        # The in-place reinstall is refused on Windows; these tests are about
        # the reinstall, so pin the platform rather than inherit the runner's.
        monkeypatch.setattr(updater.sys, "platform", "linux")
        return calls

    def test_pinned_receipt_reinstalls_instead_of_upgrading(self, runner, monkeypatch):
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "==0.9.0", "python": "3.12"},
        )
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.11.0")

        ok, _message = updater.update_via_method("uv")

        assert ok is True
        assert runner == [
            ["uv", "tool", "install", "--force", "--python", "3.12", "zotero-mcp-server[all]@latest"],
        ]

    def test_noop_upgrade_falls_through_to_reinstall(self, runner, monkeypatch):
        """Exit 0 from `uv tool upgrade` is not evidence anything changed."""
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "", "python": "3.12"},
        )
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.9.0")

        updater.update_via_method("uv")

        assert runner == [
            ["uv", "tool", "upgrade", "zotero-mcp-server"],
            ["uv", "tool", "install", "--force", "--python", "3.12", "zotero-mcp-server[all]@latest"],
        ]

    def test_effective_upgrade_stops_there(self, runner, monkeypatch):
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "", "python": "3.12"},
        )
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.11.0")

        ok, _message = updater.update_via_method("uv")

        assert ok is True
        assert runner == [["uv", "tool", "upgrade", "zotero-mcp-server"]]


class TestUpdateOutcomePolicy:
    """Equality, downgrade and 'newer but not latest' are three different outcomes."""

    def test_downgrade_is_a_failure(self, update_scaffold, monkeypatch):
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.8.0")

        result = updater.update_zotero_mcp()

        assert result["success"] is False
        assert "older" in result["message"]

    def test_newer_but_not_latest_is_success_and_says_so(self, update_scaffold, monkeypatch):
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.10.0")

        result = updater.update_zotero_mcp()

        assert result["success"] is True
        assert "0.10.0" in result["message"] and "0.11.0" in result["message"]

    def test_force_reinstall_of_latest_is_success(self, update_scaffold, monkeypatch):
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.11.0")
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.11.0")

        result = updater.update_zotero_mcp(force=True)

        assert result["success"] is True


class TestUvToolReceiptPolicy:
    def test_only_an_exact_pin_counts(self):
        assert updater._is_exact_pin("==0.9.0") is True
        assert updater._is_exact_pin("===0.9.0") is True
        assert updater._is_exact_pin("") is False
        assert updater._is_exact_pin(">=0.9,<1") is False
        assert updater._is_exact_pin("==0.9.*") is False

    def test_malformed_receipt_raises(self, tmp_path, monkeypatch):
        """A receipt that exists but cannot be read must not be mistaken for 'no pin'."""
        tool_dir = tmp_path / "zotero-mcp-server"
        tool_dir.mkdir()
        (tool_dir / "uv-receipt.toml").write_text("[tool\nthis is not toml")
        monkeypatch.setattr(updater, "_uv_tool_dir", lambda: tmp_path)

        with pytest.raises(updater.ReceiptError):
            updater._read_uv_tool_receipt()

    def test_tool_env_python(self, tmp_path):
        env = tmp_path / "zotero-mcp-server"
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python").write_text("")
        (env / "Scripts").mkdir()
        (env / "Scripts" / "python.exe").write_text("")

        python = updater._uv_tool_python(tmp_path)

        assert python in {env / "bin" / "python", env / "Scripts" / "python.exe"}
        assert python.exists()

    def test_tool_env_python_missing_is_none(self, tmp_path):
        assert updater._uv_tool_python(tmp_path) is None


class TestInstalledVersionProbe:
    def test_probe_runs_the_given_interpreter(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="0.11.0\n", stderr="")

        monkeypatch.setattr(updater.subprocess, "run", fake_run)

        assert updater.get_installed_version(python="/env/bin/python") == "0.11.0"
        assert calls[0][0] == "/env/bin/python"

    def test_probe_failure_is_none(self, monkeypatch):
        monkeypatch.setattr(
            updater.subprocess, "run",
            lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"),
        )

        assert updater.get_installed_version() is None


class TestUvToolUpdatePathEdges:
    @pytest.fixture
    def recorder(self, monkeypatch):
        """Record commands; each test decides what the fake uv does."""
        state = {"calls": [], "rc": {}, "stderr": ""}

        def fake_run(cmd, *args, **kwargs):
            state["calls"].append(list(cmd))
            rc = state["rc"].get(cmd[2] if cmd[:2] == ["uv", "tool"] else cmd[0], 0)
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=state["stderr"])

        monkeypatch.setattr(updater.subprocess, "run", fake_run)
        monkeypatch.setattr(updater, "_is_uv_tool_installation", lambda: True)
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(updater, "_uv_tool_python", lambda _d=None: None)
        monkeypatch.setattr(updater.sys, "platform", "linux")
        return state

    def test_range_specifier_is_upgraded_not_overridden(self, recorder, monkeypatch):
        """`>=0.9,<1` can move within its range; a no-op must not be 'fixed' by discarding it."""
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": ">=0.9,<1", "python": "3.12"},
        )
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.9.0")

        ok, message = updater.update_via_method("uv")

        assert ok is False
        assert ">=0.9,<1" in message
        assert recorder["calls"] == [["uv", "tool", "upgrade", "zotero-mcp-server"]]

    def test_malformed_receipt_never_reinstalls(self, recorder, monkeypatch):
        def broken():
            raise updater.ReceiptError("bad toml")

        monkeypatch.setattr(updater, "_read_uv_tool_receipt", broken)
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.9.0")

        ok, message = updater.update_via_method("uv")

        assert ok is False
        assert "receipt" in message.lower()
        assert recorder["calls"] == [["uv", "tool", "upgrade", "zotero-mcp-server"]]

    def test_unreadable_version_after_upgrade_does_not_reinstall(self, recorder, monkeypatch):
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "", "python": "3.12"},
        )
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: None)

        ok, _message = updater.update_via_method("uv")

        assert ok is False
        assert recorder["calls"] == [["uv", "tool", "upgrade", "zotero-mcp-server"]]

    def test_reinstall_failure_is_reported(self, recorder, monkeypatch):
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "==0.9.0", "python": "3.12"},
        )
        recorder["rc"]["install"] = 1
        recorder["stderr"] = "boom: no index"

        ok, message = updater.update_via_method("uv")

        assert ok is False
        assert "boom: no index" in message


class TestComposedUvUpdate:
    def test_pinned_tool_updates_end_to_end(self, monkeypatch, tmp_path):
        """Real update_zotero_mcp → real update_via_method, with uv and the disk faked together."""
        disk = {"version": "0.9.0"}
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            if cmd[:3] == ["uv", "tool", "install"]:
                disk["version"] = "0.11.0"
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[:3] == ["uv", "tool", "upgrade"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="Nothing to upgrade\n", stderr="")
            if "-c" in cmd:  # the version probe
                return subprocess.CompletedProcess(cmd, 0, stdout=disk["version"] + "\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")  # cli version check

        backup = tmp_path / "backup"
        backup.mkdir()
        monkeypatch.setattr(updater.subprocess, "run", fake_run)
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(updater, "get_latest_version", lambda: "0.11.0")
        monkeypatch.setattr(updater, "detect_installation_method", lambda: "uv")
        monkeypatch.setattr(updater, "_is_uv_tool_installation", lambda: True)
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "==0.9.0", "python": "3.12"},
        )
        monkeypatch.setattr(updater, "_uv_tool_python", lambda _d=None: None)
        monkeypatch.setattr(updater.sys, "platform", "linux")
        monkeypatch.setattr(updater, "backup_configurations", lambda: backup)
        monkeypatch.setattr(updater, "restore_configurations", lambda _b: True)

        result = updater.update_zotero_mcp()

        assert result["success"] is True
        assert result["installed_version"] == "0.11.0"
        assert "0.9.0" in result["message"] and "0.11.0" in result["message"]
        assert ["uv", "tool", "install", "--force", "--python", "3.12", "zotero-mcp-server[all]@latest"] in calls
        assert ["uv", "tool", "upgrade", "zotero-mcp-server"] not in calls


class TestSecondReviewRound:
    def test_probe_reads_distribution_metadata_not_the_module(self, monkeypatch):
        """Importing the package runs its __init__; metadata is what the installer wrote."""
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="0.11.0\n", stderr="")

        monkeypatch.setattr(updater.subprocess, "run", fake_run)

        updater.get_installed_version()

        assert "importlib.metadata" in calls[0][-1]
        assert "zotero-mcp-server" in calls[0][-1]
        assert "from zotero_mcp" not in calls[0][-1]

    def test_receipt_name_is_matched_normalized(self, tmp_path, monkeypatch):
        tool_dir = tmp_path / "zotero-mcp-server"
        tool_dir.mkdir()
        (tool_dir / "uv-receipt.toml").write_text(
            '[tool]\n'
            'requirements = [{ name = "Zotero_MCP_Server", extras = ["all"], specifier = "==0.9.0" }]\n'
            'python = "3.12"\n'
        )
        monkeypatch.setattr(updater, "_uv_tool_dir", lambda: tmp_path)

        assert updater._read_uv_tool_receipt()["specifier"] == "==0.9.0"

    def test_windows_never_reinstalls_in_place(self, monkeypatch):
        """`uv tool install --force` recreates the env that holds the running python.exe."""
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(updater.subprocess, "run", fake_run)
        monkeypatch.setattr(updater.sys, "platform", "win32")
        monkeypatch.setattr(updater, "_is_uv_tool_installation", lambda: True)
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(updater, "_uv_tool_python", lambda _d=None: None)
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "==0.9.0", "python": "3.12"},
        )

        ok, message = updater.update_via_method("uv")

        assert ok is False
        assert "zotero-mcp-server[all]@latest" in message
        assert calls == []


class TestThirdReviewRound:
    """Round two of review: `-I` hid `pip install --user` installs, and smaller gaps."""

    @pytest.fixture
    def capture_run(self, monkeypatch):
        seen: dict[str, Any] = {}

        def fake_run(cmd, *args, **kwargs):
            seen["cmd"] = list(cmd)
            seen["kwargs"] = kwargs
            return subprocess.CompletedProcess(cmd, 0, stdout="0.11.0\n", stderr="")

        monkeypatch.setattr(updater.subprocess, "run", fake_run)
        return seen

    def test_probe_keeps_user_site_and_ignores_env_and_cwd(self, capture_run):
        """`-I` implies `-s`, which hides a `pip install --user` install. `-E` plus a
        neutral working directory blocks PYTHONPATH and cwd shadowing without that."""
        updater.get_installed_version(python="/env/bin/python")

        assert "-I" not in capture_run["cmd"]
        assert "-E" in capture_run["cmd"]
        assert Path(capture_run["kwargs"]["cwd"]) == Path("/env/bin")

    def test_verify_uses_the_given_interpreter(self, capture_run):
        ok, _message = updater.verify_installation(python="/env/bin/python")

        assert ok is True
        assert capture_run["cmd"][0] == "/env/bin/python"
        assert "-I" not in capture_run["cmd"]
        assert "-E" in capture_run["cmd"]
        assert Path(capture_run["kwargs"]["cwd"]) == Path("/env/bin")

    def test_uv_tool_dir_failure_is_a_receipt_error(self, monkeypatch):
        monkeypatch.setattr(updater, "_uv_tool_dir", lambda: None)

        with pytest.raises(updater.ReceiptError):
            updater._read_uv_tool_receipt()

    def test_receipt_name_normalization_handles_dots_and_runs(self, tmp_path, monkeypatch):
        tool_dir = tmp_path / "zotero-mcp-server"
        tool_dir.mkdir()
        (tool_dir / "uv-receipt.toml").write_text(
            '[tool]\nrequirements = [{ name = "zotero.mcp__server", specifier = "==0.9.0" }]\n'
        )
        monkeypatch.setattr(updater, "_uv_tool_dir", lambda: tmp_path)

        assert updater._read_uv_tool_receipt()["specifier"] == "==0.9.0"

    def test_windows_message_quotes_a_python_path_with_spaces(self, monkeypatch):
        monkeypatch.setattr(updater.sys, "platform", "win32")
        receipt = {"extras": ["all"], "specifier": "==0.9.0", "python": r"C:\Users\First Last\py.exe"}

        message = updater._windows_reinstall_message(receipt)

        assert '"C:\\Users\\First Last\\py.exe"' in message

    def test_windows_noop_upgrade_never_reinstalls(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="Nothing to upgrade\n", stderr="")

        monkeypatch.setattr(updater.subprocess, "run", fake_run)
        monkeypatch.setattr(updater.sys, "platform", "win32")
        monkeypatch.setattr(updater, "_is_uv_tool_installation", lambda: True)
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(updater, "_uv_tool_python", lambda _d=None: None)
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.9.0")
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": "", "python": "3.12"},
        )

        ok, message = updater.update_via_method("uv")

        assert ok is False
        assert "zotero-mcp-server[all]@latest" in message
        assert calls == [["uv", "tool", "upgrade", "zotero-mcp-server"]]

    def test_tool_env_python_windows_layout(self, tmp_path):
        env = tmp_path / "zotero-mcp-server"
        (env / "Scripts").mkdir(parents=True)
        (env / "Scripts" / "python.exe").write_text("")

        assert updater._uv_tool_python(tmp_path) == env / "Scripts" / "python.exe"

    def test_probe_python_selection(self, tmp_path, monkeypatch):
        env = tmp_path / "zotero-mcp-server"
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python").write_text("")
        monkeypatch.setattr(updater, "_uv_tool_dir", lambda: tmp_path)
        monkeypatch.setattr(updater, "_is_uv_tool_installation", lambda: True)

        assert updater._probe_python("uv") == env / "bin" / "python"
        assert updater._probe_python("pip") is None
        assert updater._probe_python("pipx") is None
        assert updater._probe_python("conda") is None

    def test_range_constraint_that_can_still_move_is_upgraded(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(updater.subprocess, "run", fake_run)
        monkeypatch.setattr(updater, "_is_uv_tool_installation", lambda: True)
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(updater, "_uv_tool_python", lambda _d=None: None)
        monkeypatch.setattr(updater, "get_installed_version", lambda python=None: "0.9.1")
        monkeypatch.setattr(
            updater, "_read_uv_tool_receipt",
            lambda: {"extras": ["all"], "specifier": ">=0.9,<0.10", "python": "3.12"},
        )

        ok, _message = updater.update_via_method("uv")

        assert ok is True
        assert calls == [["uv", "tool", "upgrade", "zotero-mcp-server"]]


class TestCliOutput:
    def test_success_line_prints_the_installed_version(self, monkeypatch, capsys):
        """The symptom was on the terminal; guard the line that showed the wrong version."""
        from zotero_mcp import cli

        monkeypatch.setattr(
            updater, "update_zotero_mcp",
            lambda **kwargs: {
                "success": True, "current_version": "0.9.0", "latest_version": "0.12.0",
                "installed_version": "0.11.0", "method": "uv", "message": "Successfully updated from 0.9.0 to 0.11.0",
            },
        )
        monkeypatch.setattr(cli.sys, "argv", ["zotero-mcp", "update"])

        cli.main()

        out = capsys.readouterr().out
        assert "Version: 0.9.0 → 0.11.0" in out
        assert "0.12.0" not in out

    def test_up_to_date_line_falls_back_to_the_current_version(self, monkeypatch, capsys):
        """No install ran, so there is no installed_version to show; never print 'None'."""
        from zotero_mcp import cli

        monkeypatch.setattr(
            updater, "update_zotero_mcp",
            lambda **kwargs: {
                "success": True, "current_version": "0.11.0", "latest_version": "0.11.0",
                "installed_version": None, "method": None, "message": "Already up to date (version 0.11.0)",
            },
        )
        monkeypatch.setattr(cli.sys, "argv", ["zotero-mcp", "update"])

        cli.main()

        out = capsys.readouterr().out
        assert "Version: 0.11.0 → 0.11.0" in out
        assert "→ None" not in out
