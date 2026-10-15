"""Tests for server lifespan — verifies startup is non-blocking."""

import asyncio
import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from zotero_mcp._app import _spawn_semantic_update, server_lifespan

_SLOW_CHILD = "import time; print('child started', flush=True); time.sleep(30)"


@pytest.mark.asyncio
async def test_lifespan_yields_while_update_child_runs():
    """The lifespan yields while the update child is still running, and
    terminates it on shutdown (it resumes on the next startup)."""
    procs = []

    def spawn(cmd):
        procs.append(_spawn_semantic_update(cmd))
        return procs[-1]

    with patch("zotero_mcp._app._semantic_update_command",
               return_value=[sys.executable, "-c", _SLOW_CHILD]), \
            patch("zotero_mcp._app._spawn_semantic_update", spawn):
        async with server_lifespan(None) as ctx:
            assert ctx == {}
            await asyncio.sleep(0.1)
            assert procs, "the update child was never started"
            assert procs[0].poll() is None
    assert procs[0].wait(timeout=10) is not None


@pytest.mark.asyncio
async def test_lifespan_yields_when_update_raises():
    """Exceptions in the background update must not prevent the server
    from starting."""

    def exploding_command():
        raise RuntimeError("ChromaDB exploded")

    with patch("zotero_mcp._app._semantic_update_command", exploding_command):
        async with server_lifespan(None) as ctx:
            assert ctx == {}
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_lifespan_yields_when_no_update_due():
    """With no update due, nothing is spawned and the lifespan still yields."""
    with patch("zotero_mcp._app._semantic_update_command", return_value=None), \
            patch("zotero_mcp._app._spawn_semantic_update") as spawn:
        async with server_lifespan(None) as ctx:
            assert ctx == {}
            await asyncio.sleep(0.05)
    spawn.assert_not_called()


# ---------------------------------------------------------------------------
# #706: a due startup update must not hold up the MCP handshake
# ---------------------------------------------------------------------------

_SERVER = """
import pathlib, sys

home = pathlib.Path(sys.argv[1])
(home / ".config" / "zotero-mcp").mkdir(parents=True)
(home / ".config" / "zotero-mcp" / "config.json").write_text(
    '{"semantic_search": {"update_config":'
    ' {"auto_update": true, "update_frequency": "startup"}}}'
)
pathlib.Path.home = staticmethod(lambda: home)

# Should the update run inside the server, make its work hold the GIL in one
# native call: in the server's worker thread that kept `initialize`
# unanswered until the update finished (#706).
import zotero_mcp.semantic_search as ss
ss.create_semantic_search = lambda *a, **k: sum(range(10**10))

# Keep the real config gate, but stand in for `update-db` with a child that
# records its pid and stays busy.
import zotero_mcp._app as app
gate = app._semantic_update_command
app._semantic_update_command = lambda: gate() and [
    sys.executable, "-c",
    "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); "
    "time.sleep(60)",
    str(home / "child.pid"),
]
from zotero_mcp.server import mcp
mcp.run(transport="stdio", show_banner=False)
"""


def test_initialize_answers_while_startup_update_runs(tmp_path):
    """The server answers ``initialize`` while the update is still running (#706)."""
    pid_file = tmp_path / "home" / "child.pid"
    server = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(tmp_path / "home")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        server.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "0"}},
        }) + "\n")
        server.stdin.flush()
        sent = time.monotonic()
        reply = json.loads(server.stdout.readline())
        assert reply["id"] == 1 and "result" in reply
        # Generous for a cold start on CI; the in-process update took ~100 s.
        assert time.monotonic() - sent < 30, "initialize waited for the update"

        # The update really was running alongside, in another process.
        deadline = time.monotonic() + 30
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child_pid = int(pid_file.read_text())
        assert child_pid != server.pid

        server.stdin.close()
        server.wait(timeout=30)
    finally:
        if server.poll() is None:
            server.kill()

    # Shutting down terminated the update instead of leaving it orphaned.
    if sys.platform != "win32":
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(child_pid, 9)
            pytest.fail("the update child outlived the server")


# ---------------------------------------------------------------------------
# #485: the auto-update check must not drag ChromaDB in on every startup
# ---------------------------------------------------------------------------

# Runs in a fresh interpreter because the property under test is "was this
# module ever imported", which any earlier test in the same process would have
# already decided.
_IMPORT_PROBE = """
import json, pathlib, sys

home = pathlib.Path(sys.argv[1])
(home / ".config" / "zotero-mcp").mkdir(parents=True, exist_ok=True)
if sys.argv[2] != "__none__":
    (home / ".config" / "zotero-mcp" / "config.json").write_text(sys.argv[2])

# Patch Path.home() rather than $HOME: overriding the environment variable
# also relocates the user site-packages directory, which breaks imports that
# have nothing to do with this test.
pathlib.Path.home = staticmethod(lambda: home)

from zotero_mcp._app import _semantic_update_command

cmd = _semantic_update_command()

print(json.dumps({
    "cmd": cmd,
    "semantic_search": "zotero_mcp.semantic_search" in sys.modules,
    "chromadb": "chromadb" in sys.modules,
}))
"""


def _probe(tmp_path, raw_config):
    """Run ``_semantic_update_command`` in a subprocess; report what it imported.

    ``raw_config`` is the literal bytes of ``config.json``, or None for no file
    at all — literal rather than a dict so an unparseable config is testable.
    """
    import subprocess

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(home),
         "__none__" if raw_config is None else raw_config],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "label,raw_config",
    [
        ("no config file", None),
        ("update_config absent", '{"semantic_search": {}}'),
        (
            "auto_update false",
            '{"semantic_search": {"update_config":'
            ' {"auto_update": false, "update_frequency": "startup"}}}',
        ),
        (
            "auto_update true but frequency manual",
            '{"semantic_search": {"update_config":'
            ' {"auto_update": true, "update_frequency": "manual"}}}',
        ),
        (
            "auto_update true, daily, not yet due",
            '{"semantic_search": {"update_config": {"auto_update": true,'
            ' "update_frequency": "daily", "last_update": "2126-01-01T00:00:00"}}}',
        ),
        ("unparseable config", "{not json"),
    ],
)
def test_no_due_update_never_imports_chromadb(tmp_path, label, raw_config):
    """No update due means no ChromaDB import (#485).

    The import used to sit above the config check it was documented as being
    guarded by, so every server start paid for chromadb -> numpy. On Windows
    that import ran in the lifespan's worker thread and stalled every
    ``tools/call`` until the client timed out.
    """
    seen = _probe(tmp_path, raw_config)

    assert seen["cmd"] is None, f"{label}: an update was started"

    assert not seen["semantic_search"], f"{label}: imported zotero_mcp.semantic_search"
    assert not seen["chromadb"], f"{label}: imported chromadb"


@pytest.mark.parametrize("local", ["", "true"])
def test_due_update_runs_update_db_in_a_child(tmp_path, monkeypatch, local):
    """A due update becomes an ``update-db`` child, and the server itself
    still loads nothing heavy (#706). ``--fulltext`` follows local mode, as
    ``extract_fulltext=is_local_mode()`` did in-process."""
    monkeypatch.setenv("ZOTERO_LOCAL", local)
    seen = _probe(
        tmp_path,
        '{"semantic_search": {"update_config":'
        ' {"auto_update": true, "update_frequency": "startup"}}}',
    )

    expected = [sys.executable, "-m", "zotero_mcp.cli", "update-db"]
    assert seen["cmd"] == expected + (["--fulltext"] if local else [])
    assert not seen["semantic_search"] and not seen["chromadb"]


def test_due_update_check_is_config_only():
    """``should_update`` must stay importable without the heavy module."""
    from zotero_mcp.config_light import should_update

    assert should_update({"auto_update": True, "update_frequency": "startup"}) is True
    assert should_update({"auto_update": False, "update_frequency": "startup"}) is False
    assert should_update({"auto_update": True, "update_frequency": "manual"}) is False


def test_config_light_has_no_heavy_dependencies():
    """Guard the reason ``config_light`` exists as its own module."""
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json; import zotero_mcp.config_light; "
            "print(json.dumps(sorted(m for m in ('chromadb', 'numpy', 'torch', "
            "'zotero_mcp.semantic_search') if m in sys.modules)))",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []
