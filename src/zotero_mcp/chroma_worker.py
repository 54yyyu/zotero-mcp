"""Run ChromaDB in a child process that exits when idle, so its memory goes back to the OS.

ChromaDB's Rust core loads a collection's whole vector index the first time
anything counts or queries it, and keeps it in a process-wide cache: dropping
the client and ``SharedSystemClient.clear_system_cache()`` leave the memory in
place, and the 1.x bindings ignore ``chroma_memory_limit_bytes``. A large
full-text index is gigabytes, so an MCP server that ran one semantic search
held them for as long as it ran, often days.

In the server, :func:`zotero_mcp.chroma_client.create_chroma_client` therefore
returns a :class:`WorkerChromaClient`. It forwards every ``ChromaClient``
call to a child process (``python -m zotero_mcp.chroma_worker``) that owns
the real client, and stops that process after
``semantic_search.idle_unload_minutes`` (default 10) without a call. The next
call starts a new one, whose first query loads the index again. Everything
else a semantic search does, such as fetching the Zotero items for its hits,
stays in the server, so the active library and its clients are unaffected.

Requests and replies are length-prefixed pickles on the child's stdin and
stdout. Both ends are this package, never untrusted data. Anything the
libraries print in the child goes to stderr, so it cannot break the protocol.
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

DEFAULT_IDLE_UNLOAD_MINUTES = 10

_HEADER = struct.Struct(">Q")


def _send(stream: BinaryIO, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _receive(stream: BinaryIO) -> Any:
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        raise EOFError
    (size,) = _HEADER.unpack(header)
    payload = stream.read(size)
    if len(payload) < size:
        raise EOFError
    return pickle.loads(payload)


class WorkerExited(RuntimeError):
    """The worker process ended in the middle of a call."""


class WorkerChromaClient:
    """Stands in for :class:`ChromaClient`, running each call in the worker process.

    Methods are forwarded by name; plain attributes (``embedding_model``,
    ``persist_directory``, ...) are fetched once and cached. A method that
    returns an iterator is streamed one item per request, so the group_id
    backfill can still update documents between batches.
    """

    def __init__(self, config_path: str | None, idle_seconds: float):
        self._config_path = config_path
        self._idle_seconds = idle_seconds
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._last_used = 0.0

    # --- process lifecycle ---------------------------------------------------

    def _start(self) -> None:
        cmd = [sys.executable, "-m", "zotero_mcp.chroma_worker", self._config_path or ""]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # the server's stderr: worker logs land with the server's
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        logger.info("Started the semantic search worker (pid %s)", self._proc.pid)
        threading.Thread(
            target=self._reap_when_idle, args=(self._proc,), daemon=True, name="zmcp-chroma-reaper"
        ).start()

    def _reap_when_idle(self, proc: subprocess.Popen) -> None:
        while True:
            time.sleep(min(30.0, max(self._idle_seconds / 4, 0.05)))
            with self._lock:
                if self._proc is not proc:
                    return
                if time.monotonic() - self._last_used >= self._idle_seconds:
                    logger.info(
                        "Semantic search idle for %d s; stopping its worker to free the index memory",
                        int(self._idle_seconds),
                    )
                    self._stop_locked()
                    return

    def _stop_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.stdin.close()  # EOF: the worker returns from its loop and exits
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()
        finally:
            if proc.stdout:
                proc.stdout.close()
            self._proc = None  # only once it has exited, so worker_pid never names a live one

    def stop(self) -> None:
        """Stop the worker now, if one is running. The next call starts a new one."""
        with self._lock:
            self._stop_locked()

    @property
    def worker_pid(self) -> int | None:
        """The running worker's pid, or None between idle unloads."""
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def _request(self, message: tuple) -> Any:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                self._start()
            proc = self._proc
            try:
                _send(proc.stdin, message)
                status, value = _receive(proc.stdout)
            except (EOFError, BrokenPipeError, OSError) as e:
                self._stop_locked()
                raise WorkerExited("the semantic search worker exited during a call") from e
            finally:
                self._last_used = time.monotonic()
        if status == "error":
            raise value
        return status, value

    # --- forwarding ----------------------------------------------------------

    def _call(self, name: str, *args, **kwargs):
        status, value = self._request(("call", name, args, kwargs))
        if status == "iter":
            return self._iterate(value)
        return value

    def _iterate(self, iterator_id: int):
        while True:
            status, value = self._request(("next", iterator_id))
            if status == "done":
                return
            yield value

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        status, value = self._request(("getattr", name))
        if status == "callable":
            value = functools.partial(self._call, name)
        self.__dict__[name] = value
        return value


# --- which processes use a worker ------------------------------------------------

# Only the long-running server opts in: a CLI command exits when it is done,
# which frees the index anyway, and the worker itself must open the real client.
_enabled = False
_shared: dict[tuple[str | None, float], WorkerChromaClient] = {}
_shared_lock = threading.Lock()


def enable_worker_mode() -> None:
    """Make :func:`shared_client` hand out worker-backed clients (the MCP server does this)."""
    global _enabled
    _enabled = True


def shared_client(config_path: str | None, idle_minutes: float) -> WorkerChromaClient | None:
    """The process's worker-backed client for this config, or None to use ChromaDB in-process.

    One worker serves every tool call: ``create_chroma_client`` runs once per
    call, and a worker per call would load the index each time.
    """
    if not _enabled or idle_minutes <= 0:
        return None
    key = (config_path, float(idle_minutes))
    with _shared_lock:
        client = _shared.get(key)
        if client is None:
            client = _shared[key] = WorkerChromaClient(config_path, idle_seconds=idle_minutes * 60)
        return client


def stop_all() -> None:
    """Stop every worker this process started (server shutdown)."""
    with _shared_lock:
        clients = list(_shared.values())
    for client in clients:
        client.stop()


# --- the worker side -----------------------------------------------------------


def _serve(config_path: str | None, requests: BinaryIO, replies: BinaryIO) -> None:
    from zotero_mcp.chroma_client import create_chroma_client

    client = None
    iterators: dict[int, Any] = {}
    while True:
        try:
            message = _receive(requests)
        except EOFError:
            return
        try:
            if client is None:
                client = create_chroma_client(config_path)
            kind, name = message[0], message[1]
            if kind == "getattr":
                value = getattr(client, name)
                reply = ("callable", None) if callable(value) else ("ok", value)
            elif kind == "call":
                value = getattr(client, name)(*message[2], **message[3])
                if hasattr(value, "__next__"):
                    iterators[id(value)] = value
                    reply = ("iter", id(value))
                else:
                    reply = ("ok", value)
            elif kind == "next":
                try:
                    reply = ("ok", next(iterators[name]))
                except StopIteration:
                    del iterators[name]
                    reply = ("done", None)
            else:
                raise ValueError(f"unknown request {kind!r}")
        except BaseException as e:  # noqa: BLE001 - every failure goes back to the caller
            try:
                pickle.dumps(e)
            except Exception:
                e = RuntimeError(f"{type(e).__name__}: {e}")
            reply = ("error", e)
        try:
            _send(replies, reply)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            # A value that cannot cross the process boundary (the raw
            # ``collection``, say); pickling fails before anything is written.
            _send(replies, ("error", TypeError(f"{message[1]!r} cannot be used through the worker: {e}")))


def main() -> None:
    # Keep the protocol on a private copy of stdout and point fd 1 at stderr,
    # so a library that prints cannot corrupt a reply.
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    _serve(config_path, sys.stdin.buffer, replies)


if __name__ == "__main__":
    main()
