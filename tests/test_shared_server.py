"""Shared mode: many stdio clients served by one zotero-mcp process.

With ZOTERO_MCP_SHARED=1, ``zotero-mcp serve`` is a small stdlib-only shim
that forwards its stdin and stdout to a long-lived server over a Unix socket,
starting that server if none is running. Each connection is its own MCP
session, so what one client does must stay invisible to the others: the
active library (#739) and the Zotero API lock are per client, as they were
when every client had its own process.

The asynchronous tests run their own event loop instead of using
pytest-asyncio, which the CI test job does not install.
"""

import asyncio
import contextlib
import contextvars
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from zotero_mcp import client as _client
from zotero_mcp import shared

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="shared mode is POSIX-only")


def _in_client(scope, fn, *args):
    """Run fn as the given shared-mode client would: with that client's scope set."""
    ctx = contextvars.copy_context()

    def run():
        shared.current_client.set(scope)
        return fn(*args)

    return ctx.run(run)


@pytest.fixture
def short_dir():
    """A private directory short enough for a Unix socket path (104 bytes on macOS)."""
    path = tempfile.mkdtemp(prefix="zmcp-", dir="/tmp")
    yield Path(path)
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-client state in client.py
# ---------------------------------------------------------------------------


class TestPerClientLibraryOverride:
    def test_a_switch_is_seen_only_by_the_client_that_made_it(self, monkeypatch):
        monkeypatch.setattr(_client, "_active_library_override", {})
        a, b = shared.ClientScope(), shared.ClientScope()

        _in_client(a, _client.set_active_library, "6182688", "group")

        assert _in_client(a, _client.get_active_library) == {
            "library_id": "6182688", "library_type": "group",
        }
        assert _in_client(a, _client.get_active_group_id) == 6182688
        assert _in_client(b, _client.get_active_library) == {}
        assert _in_client(b, _client.get_active_group_id) == 0
        # Nor does it touch the process-wide override that plain stdio uses.
        assert _client.get_active_library() == {}

    def test_get_zotero_client_uses_the_callers_library(self, monkeypatch):
        monkeypatch.setattr(_client, "_active_library_override", {})
        monkeypatch.setenv("ZOTERO_LOCAL", "true")
        monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
        a, b = shared.ClientScope(), shared.ClientScope()
        _in_client(a, _client.set_active_library, "42", "group")

        za = _in_client(a, _client.get_zotero_client)
        zb = _in_client(b, _client.get_zotero_client)

        assert (za.library_type, za.library_id) == ("groups", "42")
        assert (zb.library_type, zb.library_id) == ("users", "0")

    def test_clear_only_clears_the_callers_switch(self, monkeypatch):
        monkeypatch.setattr(_client, "_active_library_override", {})
        a, b = shared.ClientScope(), shared.ClientScope()
        _in_client(a, _client.set_active_library, "1", "group")
        _in_client(b, _client.set_active_library, "2", "group")

        _in_client(a, _client.clear_active_library)

        assert _in_client(a, _client.get_active_library) == {}
        assert _in_client(b, _client.get_active_library)["library_id"] == "2"

    def test_without_a_client_scope_the_override_is_process_wide(self, monkeypatch):
        """Plain stdio and the CLI keep using the module-level dict."""
        monkeypatch.setattr(_client, "_active_library_override", {})

        _client.set_active_library("7", "group")

        assert _client._active_library_override == {"library_id": "7", "library_type": "group"}


class TestPerClientApiLock:
    def _hold_in_thread(self, scope):
        """Take the API lock as `scope` on another thread; return the release event."""
        held, release = threading.Event(), threading.Event()

        def holder():
            with _client.zotero_api_lock():
                held.set()
                release.wait(timeout=10)

        ctx = contextvars.copy_context()
        ctx.run(shared.current_client.set, scope)
        t = threading.Thread(target=ctx.run, args=(holder,))
        t.start()
        assert held.wait(timeout=5)
        return release, t

    def test_one_clients_long_call_does_not_block_another_client(self, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.3")
        a, b = shared.ClientScope(), shared.ClientScope()
        release, t = self._hold_in_thread(a)
        try:
            def take():
                with _client.zotero_api_lock():
                    return "ran"

            started = time.monotonic()
            assert _in_client(b, take) == "ran"
            assert time.monotonic() - started < 0.2
        finally:
            release.set()
            t.join()

    def test_one_clients_calls_still_queue_behind_each_other(self, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.3")
        a = shared.ClientScope()
        release, t = self._hold_in_thread(a)
        try:
            def take():
                with _client.zotero_api_lock():
                    return "ran"

            with pytest.raises(_client.ZoteroApiBusyError):
                _in_client(a, take)
        finally:
            release.set()
            t.join()


class TestPerClientLocalWriteKey:
    def test_a_single_use_key_belongs_to_the_client_granted_it(self, monkeypatch):
        """Another client's write must not consume it, nor its clear remove it."""
        monkeypatch.delenv("ZOTERO_LOCAL_API_KEY", raising=False)
        a, b = shared.ClientScope(), shared.ClientScope()
        a.local_write_state.update({"key": "once", "server_id": "s1", "remember": False})

        assert _in_client(a, _client.get_local_write_credentials) == ("once", "s1", "session")
        assert _in_client(b, _client.get_local_write_credentials) == (None, None, None)
        _in_client(b, _client.clear_local_write_credentials)
        assert _in_client(a, _client.get_local_write_credentials)[0] == "once"
        # The process-wide state of a plain stdio server is untouched.
        assert _client.get_local_write_credentials() == (None, None, None)


# ---------------------------------------------------------------------------
# Which server a client may share
# ---------------------------------------------------------------------------


class TestDaemonKey:
    BASE = {"PATH": "/usr/bin", "HOME": "/home/u", "ZOTERO_LOCAL": "true"}

    def test_unrelated_variables_do_not_split_clients(self):
        other = dict(self.BASE, PATH="/opt/bin", TERM="xterm", CLAUDE_CODE_SESSION="x")
        assert shared.daemon_key(self.BASE) == shared.daemon_key(other)

    @pytest.mark.parametrize("name,value", [
        ("ZOTERO_LIBRARY_ID", "123"),
        ("ZOTERO_MCP_TOOLSETS", "none"),
        ("ZOTERO_SOMETHING_ADDED_LATER", "1"),
        ("OPENAI_API_KEY", "sk"),
        ("HTTPS_PROXY", "http://proxy:3128"),
        ("HOME", "/home/other"),
    ])
    def test_behaviour_changing_variables_get_their_own_server(self, name, value):
        assert shared.daemon_key(self.BASE) != shared.daemon_key(dict(self.BASE, **{name: value}))

    def test_shared_mode_settings_themselves_do_not_split_clients(self):
        tuned = dict(self.BASE, ZOTERO_MCP_SHARED="1", ZOTERO_MCP_SHARED_IDLE="5")
        assert shared.daemon_key(self.BASE) == shared.daemon_key(tuned)

    def test_another_version_gets_its_own_server(self, monkeypatch):
        before = shared.daemon_key(self.BASE)
        monkeypatch.setattr(shared, "__version__", "0.0.0-other")
        assert shared.daemon_key(self.BASE) != before

    def test_another_install_gets_its_own_server(self, monkeypatch):
        before = shared.daemon_key(self.BASE)
        monkeypatch.setattr(sys, "executable", "/elsewhere/bin/python")
        assert shared.daemon_key(self.BASE) != before


class TestRequested:
    @pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("0", False), ("", False)])
    def test_flag(self, monkeypatch, value, expected):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(shared, "_fastmcp_major", lambda: 4)
        monkeypatch.setenv("ZOTERO_MCP_SHARED", value)
        assert shared.requested() is expected

    def test_windows_falls_back_to_a_private_server(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("ZOTERO_MCP_SHARED", "1")
        assert shared.requested() is False
        assert "not supported on Windows" in capsys.readouterr().err

    def test_fastmcp_2_falls_back_to_a_private_server(self, monkeypatch, capsys):
        """fastmcp 2.x runs sync tools on the event loop: one slow call would
        stall every client of a shared server."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(shared, "_fastmcp_major", lambda: 2)
        monkeypatch.setenv("ZOTERO_MCP_SHARED", "1")
        assert shared.requested() is False
        assert "needs fastmcp 3" in capsys.readouterr().err


def test_the_shim_reads_dotenv_before_choosing_local_mode(monkeypatch):
    """A private server imports client.py, and with it .env, before
    setup_zotero_environment() decides on local mode; the shim must too,
    or an API key kept in .env is lost to the shared server."""
    import dotenv

    from zotero_mcp import cli

    for name in ("ZOTERO_API_KEY", "ZOTERO_LOCAL", "ZOTERO_LIBRARY_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOTERO_NO_CLAUDE", "1")
    monkeypatch.setattr(cli, "load_standalone_env_vars", lambda: {})
    monkeypatch.setattr(dotenv, "load_dotenv",
                        lambda *a, **k: os.environ.setdefault("ZOTERO_API_KEY", "from-dotenv"))
    monkeypatch.setattr(shared, "requested", lambda: True)
    seen = {}

    def fake_shim():
        seen.update(os.environ)
        return 0

    monkeypatch.setattr(shared, "run_shim", fake_shim)
    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "serve"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert seen["ZOTERO_API_KEY"] == "from-dotenv"
    assert "ZOTERO_LOCAL" not in seen


@posix_only
class TestSocketDir:
    def test_refuses_a_directory_other_users_can_enter(self, monkeypatch, short_dir):
        os.chmod(short_dir, 0o755)
        monkeypatch.setenv("ZOTERO_MCP_SHARED_DIR", str(short_dir))
        with pytest.raises(shared.SharedUnavailable, match="private"):
            shared.socket_path()

    def test_refuses_a_path_too_long_for_a_unix_socket(self, monkeypatch, tmp_path):
        deep = tmp_path / ("x" * 120)
        monkeypatch.setenv("ZOTERO_MCP_SHARED_DIR", str(deep))
        with pytest.raises(shared.SharedUnavailable, match="too long"):
            shared.socket_path()

    def test_creates_the_directory_private(self, monkeypatch, short_dir):
        target = short_dir / "sub"
        monkeypatch.setenv("ZOTERO_MCP_SHARED_DIR", str(target))
        path = shared.socket_path()
        assert path.parent == target
        assert target.stat().st_mode & 0o777 == 0o700


# ---------------------------------------------------------------------------
# The shim's byte pump
# ---------------------------------------------------------------------------


class _FakeDaemon:
    """Stands in for the daemon Popen in _start_daemon."""

    def __init__(self, returncode, calls):
        self.returncode = returncode
        self._calls = calls

    def __call__(self, *args, **kwargs):
        self._calls.append(kwargs)
        return self

    def poll(self):
        return self.returncode

    def kill(self):
        pass


@posix_only
class TestStartingTheServer:
    def test_the_server_does_not_run_in_the_clients_directory(self, monkeypatch, short_dir):
        """`python -m` imports from the working directory; a client's project
        checkout is no place for the shared server to import from."""
        calls = []
        monkeypatch.setattr(shared.subprocess, "Popen", _FakeDaemon(1, calls))
        with pytest.raises(shared.SharedUnavailable, match="exited with status 1"):
            shared._start_daemon(short_dir / "k.sock")
        assert calls[0]["cwd"] == str(short_dir)
        assert calls[0]["start_new_session"] is True

    def test_waits_for_a_server_that_another_client_started(self, monkeypatch, short_dir):
        """A server that finds another already listening exits 75; the client
        connects to the one that is listening instead of giving up."""
        path = short_dir / "k.sock"
        calls = []
        monkeypatch.setattr(shared.subprocess, "Popen", _FakeDaemon(shared._ALREADY_SERVED, calls))
        srv = socket.socket(socket.AF_UNIX)

        def listen_soon():
            time.sleep(0.3)
            srv.bind(str(path))
            srv.listen(1)

        t = threading.Thread(target=listen_soon)
        t.start()
        sock = shared._start_daemon(path)
        t.join()
        sock.close()
        srv.close()

    def test_a_second_server_for_the_same_socket_exits(self, short_dir):
        path = short_dir / "k.sock"
        srv = socket.socket(socket.AF_UNIX)
        srv.bind(str(path))
        srv.listen(1)
        try:
            with pytest.raises(SystemExit) as exit_info:
                shared.serve_daemon(None, path)
            assert exit_info.value.code == shared._ALREADY_SERVED
        finally:
            srv.close()

    def test_any_os_error_falls_back_to_a_private_server(self, monkeypatch, short_dir, capsys):
        monkeypatch.setenv("ZOTERO_MCP_SHARED_DIR", str(short_dir))
        shared.socket_path().with_suffix(".lock").mkdir()  # cannot be opened as a file
        assert shared.run_shim() is None
        assert "starting a private server" in capsys.readouterr().err


@pytest.mark.parametrize("sent,safe", [
    (b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
     b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
     b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n', True),
    (b'{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{}}\n', True),
    (b'{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'
     b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{}}\n', False),
    (b'{"jsonrpc":"2.0","id":1,"method":"initia', False),  # cut off mid-line
])
def test_safe_to_replay(sent, safe):
    assert shared._safe_to_replay(sent) is safe


@posix_only
class TestPump:
    def _listener(self, short_dir, name="s.sock"):
        path = str(short_dir / name)
        srv = socket.socket(socket.AF_UNIX)
        srv.bind(path)
        srv.listen(4)
        return srv, path

    def _connector(self, path):
        def connect():
            s = socket.socket(socket.AF_UNIX)
            s.connect(path)
            return s
        return connect

    def test_replays_to_a_new_server_if_the_first_closes_before_answering(self, short_dir):
        """A server retiring as a client connects must not cost that client its session."""
        srv, path = self._listener(short_dir)
        in_r, in_w = os.pipe()
        out_r, out_w = os.pipe()
        connect = self._connector(path)
        result = {}
        pump = threading.Thread(
            target=lambda: result.setdefault("rc", shared._pump(connect(), connect, in_r, out_w)))
        pump.start()

        os.write(in_w, b'{"jsonrpc":"2.0","id":1,"method":"initialize"}\n')
        first, _ = srv.accept()
        assert first.recv(4096).startswith(b'{"jsonrpc"')
        first.close()  # retires without answering

        second, _ = srv.accept()
        second.settimeout(5)
        assert second.recv(4096) == b'{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'
        second.sendall(b'{"jsonrpc":"2.0","id":1,"result":{}}\n')
        assert os.read(out_r, 4096) == b'{"jsonrpc":"2.0","id":1,"result":{}}\n'

        os.close(in_w)  # client exits
        assert second.recv(4096) == b""
        second.close()
        pump.join(timeout=5)
        assert result["rc"] == 0
        srv.close()

    def test_does_not_replay_a_request_that_could_do_something_twice(self, short_dir, capfd):
        """A 2026-07-28 client may open with tools/call. If the server dies
        before answering, it may already have written to Zotero."""
        srv, path = self._listener(short_dir)
        srv.settimeout(0.5)
        in_r, in_w = os.pipe()
        out_r, out_w = os.pipe()
        connect = self._connector(path)
        result = {}
        pump = threading.Thread(
            target=lambda: result.setdefault("rc", shared._pump(connect(), connect, in_r, out_w)))
        pump.start()

        os.write(in_w, b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}\n')
        first, _ = srv.accept()
        first.recv(4096)
        first.close()

        pump.join(timeout=5)
        assert result["rc"] == 1
        with pytest.raises(socket.timeout):
            srv.accept()  # no second attempt
        os.close(in_w)
        srv.close()

    def test_exits_with_an_error_if_the_server_dies_mid_session(self, short_dir, capfd):
        srv, path = self._listener(short_dir)
        in_r, in_w = os.pipe()
        out_r, out_w = os.pipe()
        connect = self._connector(path)
        result = {}
        pump = threading.Thread(
            target=lambda: result.setdefault("rc", shared._pump(connect(), connect, in_r, out_w)))
        pump.start()

        os.write(in_w, b'{"id":1}\n')
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall(b'{"id":1,"result":{}}\n')
        assert os.read(out_r, 4096) == b'{"id":1,"result":{}}\n'
        conn.close()  # crash after the session started

        pump.join(timeout=5)
        assert result["rc"] == 1
        assert "shared server closed the connection" in capfd.readouterr().err
        os.close(in_w)
        srv.close()


# ---------------------------------------------------------------------------
# The daemon, in process, with tools that read client.py's per-client state
# ---------------------------------------------------------------------------


def _probe_server():
    from fastmcp import FastMCP

    lifespans = []

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(server):
        lifespans.append(1)
        yield {}

    mcp = FastMCP("probe", lifespan=lifespan)

    @mcp.tool
    def switch(library_id: str) -> str:
        _client.set_active_library(library_id, "group")
        return library_id

    @mcp.tool
    def active() -> str:
        return _client.get_active_library().get("library_id", "default")

    @mcp.tool
    def lifespan_count() -> int:
        return len(lifespans)

    return mcp


def _client_for(command, args, env=None, mode="auto"):
    """A client that ends its server process on exit (StdioTransport keeps it by default)."""
    import inspect

    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    transport = StdioTransport(command, args, env=env, keep_alive=False)
    if "mode" in inspect.signature(Client).parameters:
        return Client(transport, mode=mode)
    # fastmcp < 4 speaks only the handshake era.
    return Client(transport)


def _shim_client(path, mode):
    code = "import sys; from zotero_mcp import shared; sys.exit(shared.attach(sys.argv[1]))"
    return _client_for(sys.executable, ["-c", code, str(path)], mode=mode)


async def _wait_for_socket(path):
    for _ in range(100):
        if path.exists():
            return
        await asyncio.sleep(0.05)


@posix_only
@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_two_clients_on_one_daemon_keep_their_own_library(short_dir, monkeypatch, mode):
    """Both protocol eras: on 2026-07-28 connections ctx.session_id changes per
    request, so the scope has to come from the connection, not the session."""
    monkeypatch.setattr(_client, "_active_library_override", {})
    path = short_dir / "probe.sock"

    async def scenario():
        task = asyncio.create_task(shared.serve(_probe_server(), path, idle_timeout=60))
        await _wait_for_socket(path)
        try:
            async with _shim_client(path, mode) as a, _shim_client(path, mode) as b:
                await a.call_tool("switch", {"library_id": "111"})
                assert (await a.call_tool("active")).data == "111"
                assert (await a.call_tool("active")).data == "111"
                assert (await b.call_tool("active")).data == "default"
                await b.call_tool("switch", {"library_id": "222"})
                assert (await a.call_tool("active")).data == "111"
                assert (await b.call_tool("active")).data == "222"
                assert (await b.call_tool("lifespan_count")).data == 1
            assert oct(path.stat().st_mode & 0o777) == oct(0o600)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
    assert _client.get_active_library() == {}


@posix_only
def test_daemon_exits_when_idle_and_removes_its_socket(short_dir):
    path = short_dir / "idle.sock"

    async def scenario():
        # Long enough for the shim below to connect before the server gives up.
        task = asyncio.create_task(shared.serve(_probe_server(), path, idle_timeout=2.0))
        await _wait_for_socket(path)
        async with _shim_client(path, "auto") as a:
            await asyncio.sleep(3.0)  # an open connection keeps it alive
            assert (await a.call_tool("active")).data == "default"
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())
    assert not path.exists()


@posix_only
def test_an_idle_timeout_of_zero_means_never_exit(short_dir):
    path = short_dir / "forever.sock"

    async def scenario():
        task = asyncio.create_task(shared.serve(_probe_server(), path, idle_timeout=0))
        await _wait_for_socket(path)
        await asyncio.sleep(1.5)
        assert not task.done()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# End to end: the real `zotero-mcp serve`, discovery and lifecycle
# ---------------------------------------------------------------------------


def _serve_env(home, shared_dir, **extra):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "ZOTERO_NO_CLAUDE": "1",
        "ZOTERO_LOCAL": "true",
        # Nothing listens here: these tests never reach Zotero.
        "ZOTERO_LOCAL_PORT": "1",
        "ZOTERO_MCP_SHARED": "1",
        "ZOTERO_MCP_SHARED_DIR": str(shared_dir),
    }
    env.update(extra)
    return env


def _serve_client(env):
    return _client_for(sys.executable, ["-m", "zotero_mcp.cli", "serve"], env=env)


def _daemon_pids(shared_dir):
    out = subprocess.run(["pgrep", "-f", "--", f"--shared-daemon {shared_dir}"],
                         capture_output=True, text=True).stdout
    return sorted(int(p) for p in out.split())


# Below fastmcp 3 shared mode declines, and the server itself does not start
# (its toolsets call FastMCP.disable, added in 3.0).
needs_fastmcp3 = pytest.mark.skipif(shared._fastmcp_major() < 3, reason="needs fastmcp 3")


@posix_only
@needs_fastmcp3
@pytest.mark.timeout(60)
def test_clients_with_the_same_config_share_one_server(tmp_path, short_dir):
    env = _serve_env(tmp_path, short_dir, ZOTERO_MCP_SHARED_IDLE="300")

    async def scenario():
        async with _serve_client(env) as a, _serve_client(env) as b:
            tools_a = {t.name for t in await a.list_tools()}
            tools_b = {t.name for t in await b.list_tools()}
            assert "zotero_switch_library" in tools_a
            assert tools_a == tools_b
            assert len(_daemon_pids(short_dir)) == 1
            assert len(list(short_dir.glob("*.sock"))) == 1

            # A different library is a different configuration: its own server.
            other = dict(env, ZOTERO_LIBRARY_ID="99")
            async with _serve_client(other) as c:
                await c.list_tools()
                assert len(_daemon_pids(short_dir)) == 2
                assert len(list(short_dir.glob("*.sock"))) == 2

    try:
        asyncio.run(scenario())
    finally:
        for pid in _daemon_pids(short_dir):
            os.kill(pid, signal.SIGTERM)


@posix_only
@needs_fastmcp3
@pytest.mark.timeout(60)
def test_concurrent_first_clients_start_one_server(tmp_path, short_dir):
    env = _serve_env(tmp_path, short_dir, ZOTERO_MCP_SHARED_IDLE="300")

    async def scenario():
        clients = [_serve_client(env) for _ in range(4)]
        async with clients[0], clients[1], clients[2], clients[3]:
            await asyncio.gather(*(c.list_tools() for c in clients))
            assert len(_daemon_pids(short_dir)) == 1

    try:
        asyncio.run(scenario())
    finally:
        for pid in _daemon_pids(short_dir):
            os.kill(pid, signal.SIGTERM)


@posix_only
@needs_fastmcp3
@pytest.mark.timeout(60)
def test_unusable_socket_dir_falls_back_to_a_private_server(tmp_path, short_dir):
    os.chmod(short_dir, 0o777)
    env = _serve_env(tmp_path, short_dir)

    async def scenario():
        async with _serve_client(env) as a:
            assert "zotero_switch_library" in {t.name for t in await a.list_tools()}

    asyncio.run(scenario())
    assert _daemon_pids(short_dir) == []
    os.chmod(short_dir, 0o700)
