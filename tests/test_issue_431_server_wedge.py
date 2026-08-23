"""Regression tests for #431: one hung tool call wedged the whole server.

Reported on Windows: ``zotero_get_pdf_outline`` never returned, and from then
on *every* other MCP tool timed out with no recovery. Separately, concurrent
tool calls produced "Connection closed" and orphan server processes.

Three mechanisms are covered here, all testable off Windows:

* ``get_pdf_outline`` held the process-global Zotero API lock across the
  outline extraction, so one stuck PDF blocked every other tool behind it.
  The lock now covers the Zotero API work only.
* ``_extract_pdf_toc`` must return within a bounded time on every path, and
  must not hand the server's stdin (the MCP pipe under the stdio transport)
  to a child that can outlive it.
* ``suppress_stdout`` swapped the global ``sys.stdout`` without any
  synchronization, so two concurrent users could leave it pointing at a
  closed devnull.
"""

import subprocess
import sys
import threading
import time

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import server
from zotero_mcp import utils as utils_module
from zotero_mcp.tools import write as write_tools
from zotero_mcp.utils import suppress_stdout

ATTACHMENT_KEY = "ATT00001"
PARENT_KEY = "PAR00001"

# The lock is probed through write.py's own bindings rather than a directly
# imported one: a few test modules reinstall a freshly executed
# zotero_mcp.client into sys.modules at import time, so an imported lock object
# is not necessarily the object the tool under test actually takes.


def _lock_is_free(timeout=5.0):
    """Try to take the Zotero API lock from a *different* thread.

    The lock is an RLock, so a same-thread acquire always succeeds and would
    prove nothing. Requires ZOTERO_MCP_LOCK_TIMEOUT to be set short, so a held
    lock reports "busy" quickly.
    """
    result = {}

    def probe():
        try:
            with write_tools.zotero_api_lock():
                result["free"] = True
        except write_tools.ZoteroApiBusyError:
            result["free"] = False

    t = threading.Thread(target=probe)
    t.start()
    t.join(timeout=timeout)
    return result.get("free", False)


class TestOutlineDoesNotHoldTheApiLock:
    """The extraction subprocess must run with the API lock released."""

    def _setup(self, monkeypatch, tmp_path, fake_zot):
        monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.3")
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        fake_zot._children[PARENT_KEY] = [
            {
                "key": ATTACHMENT_KEY,
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                    "filename": "paper.pdf",
                    "parentItem": PARENT_KEY,
                },
            }
        ]
        client_module = write_tools._client
        monkeypatch.setattr(client_module, "get_zotero_client", lambda: fake_zot)
        monkeypatch.setattr(write_tools._utils, "is_local_mode", lambda: True)
        monkeypatch.setattr(
            client_module,
            "download_attachment_file",
            lambda *_a, **_k: client_module.AttachmentDownloadResult(
                path=pdf_path, source="Local Zotero", errors=[]
            ),
        )

    def test_lock_is_free_while_extraction_runs(
        self, monkeypatch, tmp_path, fake_zot
    ):
        """This is the wedge from #431: a slow PDF must not block other tools."""
        self._setup(monkeypatch, tmp_path, fake_zot)
        observed = {}

        def _slow_extract(*_a, **_k):
            observed["lock_free"] = _lock_is_free()
            return write_tools.TocOutcome("ok", [[1, "Intro", 1]])

        monkeypatch.setattr(write_tools, "_extract_pdf_toc", _slow_extract)

        result = server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        assert "- Intro (p. 1)" in result
        assert observed["lock_free"] is True, (
            "the Zotero API lock was still held while the outline was being "
            "extracted; one hung PDF would block every other tool"
        )

    def test_api_work_still_runs_under_the_lock(
        self, monkeypatch, tmp_path, fake_zot
    ):
        """Serialization of the actual Zotero calls is preserved."""
        self._setup(monkeypatch, tmp_path, fake_zot)
        observed = {}

        class LockProbingZotero(FakeZotero):
            def children(self, item_key, **kwargs):
                observed["lock_free_during_api"] = _lock_is_free()
                return fake_zot.children(item_key, **kwargs)

        probing = LockProbingZotero()
        probing._children = fake_zot._children
        monkeypatch.setattr(
            write_tools._client, "get_zotero_client", lambda: probing
        )
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("ok", []),
        )

        server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        assert observed["lock_free_during_api"] is False

    def test_lock_released_after_the_call(self, monkeypatch, tmp_path, fake_zot):
        self._setup(monkeypatch, tmp_path, fake_zot)
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("ok", []),
        )

        server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        assert _lock_is_free() is True


class TestExtractPdfTocIsBounded:
    """Every exit path returns; nothing waits on a pipe forever."""

    def test_hung_child_returns_promptly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            write_tools, "_TOC_CHILD_SCRIPT", "import time; time.sleep(120)"
        )

        start = time.monotonic()
        outcome = write_tools._extract_pdf_toc(
            str(tmp_path / "paper.pdf"), timeout=1
        )
        elapsed = time.monotonic() - start

        assert outcome.status == "timeout"
        assert elapsed < 15, f"timeout path took {elapsed:.1f}s"

    def test_timeout_does_not_reread_the_pipes(self, monkeypatch, tmp_path):
        """After the kill we must not re-enter communicate().

        That is what ``subprocess.run`` does, and on Windows it can block
        forever when WerFault.exe still holds the dead child's stdout/stderr
        handles: the pipes never reach EOF (#431).
        """
        calls = []

        class FakeProc:
            returncode = None
            stdout = None
            stderr = None

            def communicate(self, timeout=None):
                calls.append("communicate")
                if len(calls) == 1:
                    raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
                raise AssertionError("communicate() called again after the kill")

            def kill(self):
                calls.append("kill")

            def wait(self, timeout=None):
                calls.append(("wait", timeout))
                return -9

        monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: FakeProc())

        outcome = write_tools._extract_pdf_toc(
            str(tmp_path / "paper.pdf"), timeout=1
        )

        assert outcome.status == "timeout"
        assert calls.count("communicate") == 1
        assert "kill" in calls
        # The post-kill wait is bounded, never open-ended.
        wait_timeouts = [c[1] for c in calls if isinstance(c, tuple)]
        assert wait_timeouts and all(t is not None for t in wait_timeouts)

    def test_returns_even_if_the_child_cannot_be_reaped(
        self, monkeypatch, tmp_path
    ):
        """An unreapable child is left to the OS rather than hanging the tool."""

        class UnreapableProc:
            returncode = None
            stdout = None
            stderr = None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)

            def kill(self):
                raise OSError("access denied")

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)

        monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: UnreapableProc())

        outcome = write_tools._extract_pdf_toc(
            str(tmp_path / "paper.pdf"), timeout=1
        )

        assert outcome.status == "timeout"

    def test_child_stdin_is_devnull(self, monkeypatch, tmp_path):
        """Under stdio transport the server's stdin IS the client's pipe.

        A child that inherits it and then outlives its parent holds the
        connection open (#431).
        """
        captured = {}
        real_popen = subprocess.Popen

        def recording_popen(*args, **kwargs):
            captured.update(kwargs)
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", recording_popen)
        monkeypatch.setattr(
            write_tools,
            "_TOC_CHILD_SCRIPT",
            "import sys; sys.stdout.write("
            + repr(write_tools._TOC_SENTINEL)
            + " + '[]')",
        )

        outcome = write_tools._extract_pdf_toc(str(tmp_path / "paper.pdf"))

        assert outcome.status == "ok"
        assert captured.get("stdin") is subprocess.DEVNULL

    def test_child_script_is_valid_and_guards_windows_error_reporting(self):
        compile(write_tools._TOC_CHILD_SCRIPT, "<toc-child>", "exec")
        assert "SetErrorMode" in write_tools._TOC_CHILD_SCRIPT
        # Guarded so it cannot break the child anywhere else.
        assert "win32" in write_tools._TOC_CHILD_SCRIPT
        assert "zotero_mcp" not in write_tools._TOC_CHILD_SCRIPT

    @pytest.mark.skipif(
        sys.platform == "win32", reason="the guard is a no-op path off Windows"
    )
    def test_child_script_runs_on_this_platform(self, tmp_path):
        """The Windows guard must not raise on POSIX."""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 not really a pdf")

        outcome = write_tools._extract_pdf_toc(str(pdf), timeout=30)

        # fitz may or may not be installed here; either way the guard itself
        # must not be what fails.
        assert outcome.status in {"ok", "no_pymupdf", "error", "crashed"}
        assert "SetErrorMode" not in outcome.detail


class TestSuppressStdoutIsThreadSafe:
    """Concurrent use must not leave sys.stdout pointing at a closed file."""

    def test_concurrent_use_restores_the_real_stdout(self):
        original = sys.stdout
        errors = []
        start = threading.Event()

        def worker(delay):
            try:
                start.wait(timeout=5)
                with suppress_stdout():
                    print("swallowed")
                    time.sleep(delay)
            except Exception as exc:  # pragma: no cover - failure detail
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(d,))
            for d in (0.05, 0.15, 0.25, 0.02, 0.2)
        ]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert sys.stdout is original
        assert not sys.stdout.closed
        # Writable again: the pre-fix failure mode was a closed devnull here.
        print("stdout still works")

    def test_nested_use_survives(self):
        original = sys.stdout
        with suppress_stdout():
            inner = sys.stdout
            with suppress_stdout():
                print("swallowed")
            # Still suppressed: the inner block must not restore early.
            assert sys.stdout is inner
            assert not sys.stdout.closed
        assert sys.stdout is original
        assert not sys.stdout.closed

    def test_exception_inside_block_still_restores(self):
        original = sys.stdout
        with pytest.raises(RuntimeError):
            with suppress_stdout():
                raise RuntimeError("boom")
        assert sys.stdout is original
        assert not sys.stdout.closed
        assert utils_module._stdout_depth == 0
