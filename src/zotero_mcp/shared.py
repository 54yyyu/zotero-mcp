"""Shared mode: many stdio clients served by one zotero-mcp process.

Every stdio MCP client starts its own server, so ten Claude Code sessions,
their subagents and a Claude Desktop window mean ten or more copies of the
same ~100 MB process, each talking to the same Zotero and the same ChromaDB
index. With ``ZOTERO_MCP_SHARED=1`` in the client's environment,
``zotero-mcp serve`` instead runs as a shim:

* The **shim** (:func:`run_shim`) imports only the standard library. It finds
  the shared server for its configuration, starting one if there is none,
  and copies bytes between its stdin/stdout and a Unix socket connection
  until either side closes. It parses nothing it forwards.
* The **daemon** (:func:`serve`) is an ordinary zotero-mcp server whose
  transport is a listening socket. Each connection is served exactly as
  stdio would serve that client, by the MCP SDK's own ``stdio_server``, so
  every connection is its own MCP session. The server lifespan (schema
  refresh, semantic auto-update) runs once per daemon.

Clients with a different configuration never share a daemon: the socket is
named after a hash of the version, the interpreter and every environment
variable that changes behaviour (:func:`daemon_key`). An idle daemon exits.

State that used to be one-per-process because each client had its own
process is one-per-client here: :class:`ClientScope`, set for each
connection in :data:`current_client`, carries the active library (#739) and
the Zotero API lock. The connection is the identity because nothing in the
protocol is: on 2026-07-28 connections the MCP session id changes with every
request.

POSIX only for now: Windows has no Unix sockets in Python, and there
``ZOTERO_MCP_SHARED=1`` falls back to a private stdio server.

This module's top level must stay stdlib-only: the shim imports it, and
importing the server would cost the memory shared mode exists to save.
"""

import contextvars
import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from zotero_mcp._version import __version__

ENV_FLAG = "ZOTERO_MCP_SHARED"

#: How long a daemon with no connections waits before exiting, in seconds.
#: Long enough that a pause between sessions doesn't make the next one pay a
#: cold start. Override with ZOTERO_MCP_SHARED_IDLE.
DEFAULT_IDLE_TIMEOUT = 600.0

#: How long a shim waits for a daemon it started to accept connections.
_START_TIMEOUT = 60.0

#: sun_path is 104 bytes on macOS and 108 on Linux, including the NUL.
_MAX_SOCKET_PATH = 103

#: Exit status of a daemon that found another already serving its socket.
_ALREADY_SERVED = 75  # EX_TEMPFAIL

# Variables outside these that change what a server does. Proxy and CA
# settings matter because a daemon inherits them from whichever client
# happened to start it.
_KEY_PREFIXES = ("ZOTERO_", "OPENAI_", "GEMINI_", "OLLAMA_")
_KEY_NAMES = frozenset({
    "GOOGLE_API_KEY",
    "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
})


class SharedUnavailable(Exception):
    """No shared server can be used; the caller serves this client itself."""


class ClientScope:
    """What each client of a shared server gets its own copy of.

    These were process-wide when every client had its own process, and
    client.py reads them from here whenever a scope is set.
    """

    def __init__(self):
        self.library_override: dict[str, str] = {}
        self.api_lock = threading.RLock()
        # A key granted with a single-use "Allow" lives only here, so another
        # client's write can't consume it.
        self.local_write_state: dict = {}


#: The client being served, set for each connection by the daemon. None in a
#: plain stdio or HTTP server and in the CLI, where the process-wide state in
#: client.py applies unchanged.
current_client: contextvars.ContextVar[ClientScope | None] = contextvars.ContextVar(
    "zotero_mcp_current_client", default=None
)


def _flag_set(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _fastmcp_major() -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return int(version("fastmcp").split(".")[0])
    except (PackageNotFoundError, ValueError):
        return 0


def requested() -> bool:
    """True when this stdio server should attach to a shared one."""
    if not _flag_set(ENV_FLAG):
        return False
    if sys.platform == "win32":
        reason = "is not supported on Windows yet"
    elif _fastmcp_major() < 3:
        # fastmcp 2.x runs sync tools on the event loop, so one client's slow
        # call would stall every other client of the server.
        reason = "needs fastmcp 3 or later"
    else:
        return True
    sys.stderr.write(f"zotero-mcp: {ENV_FLAG} {reason}; starting a private server.\n")
    return False


def daemon_key(environ=None) -> str:
    """Hash of everything that decides whether two clients may share a server."""
    environ = os.environ if environ is None else environ
    parts = [__version__, sys.executable]
    parts += sorted(
        f"{name}={value}"
        for name, value in environ.items()
        if (name.startswith(_KEY_PREFIXES) or name.upper() in _KEY_NAMES)
        and not name.startswith(ENV_FLAG)
    )
    return hashlib.sha256("\0".join(parts).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _socket_dir() -> Path:
    explicit = os.environ.get(f"{ENV_FLAG}_DIR")
    if explicit:
        return Path(explicit)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "zotero-mcp"
    # On macOS gettempdir() is already per-user; elsewhere it is usually /tmp.
    return Path(tempfile.gettempdir()) / f"zotero-mcp-{os.getuid()}"


def socket_path() -> Path:
    """The socket of the shared server for this process's configuration.

    Raises SharedUnavailable when the directory is not private to this user
    (another user could otherwise talk to our server, or plant one for us to
    talk to) or the path is too long to bind.
    """
    directory = _socket_dir()
    path = directory / f"{daemon_key()}.sock"
    if len(os.fsencode(path)) > _MAX_SOCKET_PATH:
        raise SharedUnavailable(f"socket path is too long for a Unix socket: {path}")
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(directory)
    except OSError as e:
        raise SharedUnavailable(f"cannot use {directory}: {e}") from e
    if not os.path.isdir(directory) or os.path.islink(directory):
        raise SharedUnavailable(f"{directory} is not a directory")
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise SharedUnavailable(f"{directory} must be private to this user (mode 0700)")
    return path


# ---------------------------------------------------------------------------
# Shim
# ---------------------------------------------------------------------------


def run_shim() -> int | None:
    """Serve this client through the shared server.

    Returns the exit status once the client or the server closes, or None if
    no shared server could be reached, in which case nothing has been read
    from stdin yet and the caller serves the client itself.
    """
    try:
        path = socket_path()
        sock = _connect(path) or _start_daemon(path)
    except (SharedUnavailable, OSError) as e:
        sys.stderr.write(f"zotero-mcp: shared mode unavailable ({e}); starting a private server.\n")
        return None
    return _pump(sock, lambda: _connect(path) or _start_daemon(path))


def attach(path) -> int:
    """Forward stdin/stdout to the server already listening at ``path``."""
    def connect():
        sock = _connect(Path(path))
        if sock is None:
            raise SharedUnavailable(f"nothing is listening at {path}")
        return sock
    return _pump(connect(), connect)


def _connect(path: Path) -> socket.socket | None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(path))
    except (FileNotFoundError, ConnectionRefusedError):
        sock.close()
        return None
    return sock


def _start_daemon(path: Path) -> socket.socket:
    """Start the server for ``path`` unless another client just did; connect.

    The lock file makes clients that launch together start one server between
    them: each takes the lock in turn and tries to connect again before
    starting anything.
    """
    import fcntl

    log_path = path.with_suffix(".log")
    with open(path.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sock = _connect(path)
        if sock is not None:
            return sock
        # Whatever is left at the path is a dead server's socket.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        with open(log_path, "w") as log:
            # Detached, with none of this client's pipes: a server holding the
            # first client's stdout open would keep that client from ever
            # seeing its server exit (#431). Run from the private socket
            # directory, because `-m` puts the working directory on sys.path
            # and this client's (a project checkout, say) is no place to
            # import from for a server every other client will use.
            proc = subprocess.Popen(
                [sys.executable, "-m", "zotero_mcp.cli", "serve", "--shared-daemon", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(path.parent),
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + _START_TIMEOUT
        while time.monotonic() < deadline:
            sock = _connect(path)
            if sock is not None:
                return sock
            # _ALREADY_SERVED: a server whose starter gave up came up first.
            if proc.poll() is not None and proc.returncode != _ALREADY_SERVED:
                raise SharedUnavailable(
                    f"the shared server exited with status {proc.returncode}; see {log_path}"
                )
            time.sleep(0.05)
        proc.kill()
        raise SharedUnavailable(f"the shared server did not start within {_START_TIMEOUT:.0f}s; see {log_path}")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


#: Requests a server can be sent twice without doing anything twice.
_REPLAYABLE = frozenset({"initialize", "server/discover", "ping"})


def _safe_to_replay(sent: bytes) -> bool:
    """True if every request in ``sent`` only opens a session or lists things.

    A 2026-07-28 client may open with tools/call; sending that to a second
    server could repeat a write the first one made before it died.
    """
    import json

    for line in bytes(sent).splitlines():
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            return False  # a partial line: no telling what it asks for
        if not isinstance(message, dict):
            return False
        method = message.get("method")
        if "id" not in message or not isinstance(method, str):
            continue  # a notification
        if method not in _REPLAYABLE and not method.endswith("/list"):
            return False
    return True


def _pump(sock: socket.socket, reconnect, stdin_fd: int = 0, stdout_fd: int = 1) -> int:
    """Copy bytes both ways until the client or the server is done.

    Until the server's first byte arrives, everything sent is kept, so a
    server that closes without answering (one retiring just as this client
    connected) can be replaced by a fresh one and sent it all again, as long
    as nothing sent could do anything twice. After that the session lives in
    the server, and losing it ends this client.
    """
    lock = threading.Lock()
    state = {"sock": sock, "sent": bytearray(), "answered": False, "stdin_closed": False}

    def upstream():
        while True:
            try:
                data = os.read(stdin_fd, 65536)
            except OSError:
                data = b""
            with lock:
                if not data:
                    state["stdin_closed"] = True
                elif not state["answered"]:
                    state["sent"] += data
                try:
                    if data:
                        state["sock"].sendall(data)
                    else:
                        state["sock"].shutdown(socket.SHUT_WR)
                except OSError:
                    pass  # the server is gone; the loop below sees it close
            if not data:
                return

    threading.Thread(target=upstream, name="zotero-mcp-shim", daemon=True).start()

    replays = 0
    while True:
        try:
            data = state["sock"].recv(65536)
        except OSError:
            data = b""
        if data:
            # Not under the lock: upstream may be blocked in sendall while the
            # server waits for us to drain its output.
            state["answered"] = True
            try:
                _write_all(stdout_fd, data)
            except OSError:
                return 0  # the client is gone
            continue
        if state["stdin_closed"]:
            return 0
        if not state["answered"] and replays < 2 and _safe_to_replay(state["sent"]):
            replays += 1
            with lock:
                try:
                    fresh = reconnect()
                    fresh.sendall(bytes(state["sent"]))
                    if state["stdin_closed"]:
                        fresh.shutdown(socket.SHUT_WR)
                except (OSError, SharedUnavailable) as e:
                    sys.stderr.write(f"zotero-mcp: could not reach a shared server: {e}\n")
                    return 1
                state["sock"].close()
                state["sock"] = fresh
            continue
        sys.stderr.write(
            "zotero-mcp: the shared server closed the connection; "
            "restart this MCP server to reconnect.\n"
        )
        return 1


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


def idle_timeout() -> float:
    """ZOTERO_MCP_SHARED_IDLE in seconds; 0 or less means never exit when idle."""
    try:
        return float(os.environ.get(f"{ENV_FLAG}_IDLE", DEFAULT_IDLE_TIMEOUT))
    except ValueError:
        return DEFAULT_IDLE_TIMEOUT


class _SocketText:
    """The text stdin and stdout ``stdio_server`` takes, over one connection.

    Lines are split here rather than through a file object: anyio's file
    wrapper reads in a worker thread, which would hold one thread per idle
    client from the pool the tools run on.
    """

    def __init__(self, stream):
        self._stream = stream

    async def __aiter__(self):
        import anyio

        buf = bytearray()
        scanned = 0
        while True:
            try:
                buf += await self._stream.receive()
            except (anyio.EndOfStream, anyio.BrokenResourceError, anyio.ClosedResourceError):
                break
            while (end := buf.find(b"\n", scanned)) != -1:
                line = bytes(buf[: end + 1])
                del buf[: end + 1]
                scanned = 0
                yield line.decode("utf-8", errors="replace")
            scanned = len(buf)
        if buf.strip():
            yield buf.decode("utf-8", errors="replace")

    async def write(self, text: str) -> None:
        await self._stream.send(text.encode("utf-8"))

    async def flush(self) -> None:
        pass


async def serve(server, path, idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> None:
    """Serve ``server`` to every client that connects to ``path``.

    Each connection runs the way ``FastMCP.run_stdio_async`` runs its one
    client — ``stdio_server`` streams handed to the low-level server — inside
    one lifespan for the whole daemon, the pattern FastMCP's SSE transport
    uses for its connections. Returns after ``idle_timeout`` seconds with no
    connections, having removed the socket.
    """
    import logging

    import anyio
    from mcp.server.stdio import stdio_server

    try:
        from fastmcp.server.context import reset_transport, set_transport
    except ImportError:  # fastmcp releases without transport tracking
        set_transport = reset_transport = None

    logger = logging.getLogger(__name__)
    path = Path(path)
    active = 0
    last_seen = time.monotonic()
    if _fastmcp_major() < 4:
        # As fastmcp 3's run_stdio_async announces it; 4 leaves it to the SDK.
        from mcp.server.lowlevel.server import NotificationOptions

        init_options = server._mcp_server.create_initialization_options(
            notification_options=NotificationOptions(tools_changed=True)
        )
    else:
        init_options = server._mcp_server.create_initialization_options()

    async def handle(conn):
        nonlocal active, last_seen
        active += 1
        current_client.set(ClientScope())
        try:
            async with conn:
                text = _SocketText(conn)
                async with stdio_server(stdin=text, stdout=text) as (read_stream, write_stream):
                    await server._mcp_server.run(read_stream, write_stream, init_options)
        except Exception:
            # One client's failure, including leaving mid-write, is not the
            # other clients' problem.
            logger.warning("shared-mode connection ended with an error", exc_info=True)
        finally:
            active -= 1
            last_seen = time.monotonic()

    token = set_transport("stdio") if set_transport else None
    try:
        async with server._lifespan_manager():
            listener = await anyio.create_unix_listener(path)
            os.chmod(path, 0o600)
            inode = os.stat(path).st_ino
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(listener.serve, handle)
                    while True:
                        await anyio.sleep(min(1.0, idle_timeout / 4) if idle_timeout > 0 else 1.0)
                        if (
                            idle_timeout > 0
                            and active == 0
                            and time.monotonic() - last_seen >= idle_timeout
                        ):
                            break
                    # No await between the check and here, so no connection has
                    # been answered since. One accepted but unanswered is cut
                    # off, and its shim replays to a fresh server.
                    _unlink_if_ours(path, inode)
                    tg.cancel_scope.cancel()
            finally:
                _unlink_if_ours(path, inode)
                await listener.aclose()
    finally:
        if token is not None:
            reset_transport(token)


def _unlink_if_ours(path: Path, inode: int) -> None:
    try:
        if os.stat(path).st_ino == inode:
            path.unlink()
    except FileNotFoundError:
        pass


def serve_daemon(server, path) -> None:
    """Entry point of ``zotero-mcp serve --shared-daemon PATH``."""
    import anyio

    # Normally the starting client's lock means nothing else serves this
    # socket. But if that client gave up while this server was loading, the
    # next one will have started another: let whichever is listening win.
    sock = _connect(Path(path))
    if sock is not None:
        sock.close()
        sys.stderr.write(f"zotero-mcp: a shared server already listens on {path}; exiting.\n")
        sys.exit(_ALREADY_SERVED)
    sys.stderr.write(f"zotero-mcp {__version__}: shared server listening on {path}\n")
    anyio.run(serve, server, path, idle_timeout())
    sys.stderr.write("zotero-mcp: shared server idle; exiting.\n")
