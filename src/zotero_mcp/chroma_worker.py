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
``semantic_search.idle_unload_minutes`` (off by default; 10 is a good value) without a call. The next
call starts a new one, whose first query loads the index again. Everything
else a semantic search does, such as fetching the Zotero items for its hits,
stays in the server, so the active library and its clients are unaffected.

Once a worker has passages loaded, only passage activity keeps it: a search
at that level, or a call that writes or deletes passages. The passage
collection is large and slow to load again, the article collection small and
quick, so article searches alone let the worker exit and start a new one
without passages. This applies only where passages have their own
collection: on an index not yet migrated they share the article collection,
and every call keeps the worker. A worker never exits while this process
runs an index update (holds the update lock) or has an iterator open on it.

Requests and replies are length-prefixed pickles on the child's stdin and
stdout. Both ends are this package, never untrusted data. Anything the
libraries print in the child goes to stderr, so it cannot break the protocol.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
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

DEFAULT_IDLE_UNLOAD_MINUTES = 0

_HEADER = struct.Struct(">Q")

_clock = time.monotonic  # one place, so tests can move time without waiting

# Index updates running in this process, and when the last one ended. While
# one runs no worker exits: an update goes on between its calls (fetching
# items, extracting text), and a new worker would load everything again.
_updates_running = 0
_last_update_end = 0.0
_updates_lock = threading.Lock()


@contextlib.contextmanager
def update_running():
    """Keep every worker of this process alive while an index update runs."""
    global _updates_running, _last_update_end
    with _updates_lock:
        _updates_running += 1
    try:
        yield
    finally:
        with _updates_lock:
            _updates_running -= 1
            _last_update_end = _clock()


# Calls that delete passages whatever their arguments, and calls that write
# or delete the records named by ``ids``, where a ``KEY#n`` id is a passage.
_PASSAGE_CALLS = frozenset({"delete_item_chunks", "prune_item_chunks", "reset_collection"})
_CALLS_BY_ID = frozenset({
    "add_documents", "upsert_documents", "upsert_embeddings", "delete_documents", "update_metadatas",
})


def _touches_passages(name: str, args: tuple, kwargs: dict) -> bool:
    """Whether a ``ChromaClient`` call is passage activity, which keeps a worker with passages."""
    if name in _PASSAGE_CALLS:
        return True
    if name != "search" and name not in _CALLS_BY_ID:
        return False
    from zotero_mcp.chroma_client import PASSAGES, ChromaClient

    try:
        bound = inspect.signature(getattr(ChromaClient, name)).bind(None, *args, **kwargs)
    except TypeError:  # the worker will report the bad call
        return False
    if name == "search":
        return bound.arguments.get("level") == PASSAGES
    return any("#" in str(doc_id) for doc_id in bound.arguments.get("ids") or ())


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


class _EmbeddingFunctionProxy:
    """The worker's embedding function as the server's indexing code reads it.

    Indexing reads settings off ``client.embedding_function`` (how many
    requests to run in parallel, characters per token) and, when it runs
    parallel requests, calls it from its own threads. Forwarded by name the
    attribute would be a callable that runs in the worker, one request at a
    time and without those settings. So settings are read from the worker's
    function, which stays the only place a local model is loaded, and calls
    run here on a copy built from the same model name and config: embedding
    holds no index memory, and API providers are cheap to build. A provider
    that loads a local model has no parallel requests, so is never called here.

    The server's copy and the worker's have separate rate limiters, so a
    search overlapping an update can slightly exceed ``tokens_per_minute``.
    """

    def __init__(self, client: WorkerChromaClient):
        self._client = client
        self._local = None
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        value = self._client._request(("efattr", name))[1]
        self.__dict__[name] = value  # one round trip per setting, as WorkerChromaClient does
        return value

    def _function(self):
        with self._lock:
            if self._local is None:
                from zotero_mcp.embeddings import registry

                self._local = registry.create_embedding_function(
                    self._client.embedding_model, self._client.embedding_config
                )
        return self._local

    def __call__(self, documents):
        return self._function()(documents)

    def check_reachable(self) -> None:
        """The update's API check, on the server's copy (see the class doc).

        Asked of the worker's function first, so a local model is never
        loaded here: only remote providers have ``probe_max_retries``.
        """
        if getattr(self, "probe_max_retries", None) is None:
            return
        function = self._function()
        if check := getattr(type(function), "check_reachable", None):
            check(function)


class WorkerChromaClient:
    """Stands in for :class:`ChromaClient`, running each call in the worker process.

    Methods are forwarded by name; plain attributes (``embedding_model``,
    ``persist_directory``, ...) are fetched once and cached. A method that
    returns an iterator is streamed one item per request, so the group_id
    backfill can still update documents between batches.

    The first passage search in each worker gets a ``passage_load`` entry in
    its result: how long it took, which is mostly loading the passage
    collection, and whether an idle exit had freed it, so a slow search can be
    explained to the user.
    """

    def __init__(self, config_path: str | None, idle_seconds: float):
        self._config_path = config_path
        self._idle_seconds = idle_seconds
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._last_used = 0.0
        # Per worker, reset when one starts:
        self._passages_loaded = False  # from its first passage call on, only passage calls keep it
        self._last_passage_used = 0.0
        self._passage_searched = False  # its first passage search has been reported
        self._open_iterators = 0
        self._generation = 0  # which worker an iterator belongs to
        self._last_iterator_end = 0.0
        # Whether an idle exit freed loaded passages, so the next load is a reload.
        self._passages_freed = False
        self._layout_check_warned = False  # a failed separate_passages() is logged once

    # --- process lifecycle ---------------------------------------------------

    def _start(self) -> None:
        self._generation += 1
        self._passages_loaded = False
        self._passage_searched = False
        self._open_iterators = 0
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
                if self._proc is not proc or self._stop_if_idle(proc):
                    return

    def _stop_if_idle(self, proc: subprocess.Popen) -> bool:
        """Stop ``proc`` if it has been idle long enough; True if it is gone."""
        with self._lock:
            if self._proc is not proc:
                return True
            if _updates_running or self._open_iterators:
                return False
            last = self._last_passage_used if self._passages_loaded else self._last_used
            last = max(last, self._last_iterator_end, _last_update_end)
            if _clock() - last < self._idle_seconds:
                return False
            logger.info(
                "Semantic search idle for %d s%s; stopping its worker to free the index memory",
                int(self._idle_seconds),
                " (no passage search or passage update)" if self._passages_loaded else "",
            )
            self._passages_freed = self._passages_loaded
            self._stop_locked()
            return True

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

    def _request(self, message: tuple, passages: bool = False) -> Any:
        with self._lock:
            started = _clock()  # the user waits for a new worker to start, too
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                self._start()
            proc = self._proc
            first_passage_search = passages and message[1] == "search" and not self._passage_searched
            try:
                _send(proc.stdin, message)
                status, value = _receive(proc.stdout)
            except (EOFError, BrokenPipeError, OSError) as e:
                self._stop_locked()
                raise WorkerExited("the semantic search worker exited during a call") from e
            finally:
                self._last_used = _clock()
                if passages:
                    self._passages_loaded = True
                    self._last_passage_used = self._last_used
            if status == "iter":
                self._open_iterators += 1
                value = _RemoteIterator(self, value, self._generation)
            embedding_seconds = 0.0
            if message[0] == "call" and message[1] == "search" and isinstance(value, dict):
                embedding_seconds = value.pop("embedding_seconds", 0.0) or 0.0
            if first_passage_search and status != "error":
                self._passage_searched = True
                if isinstance(value, dict):
                    value["passage_load"] = {
                        # Not the query's embedding: a slow API is not a slow load.
                        "seconds": max(0.0, self._last_used - started - embedding_seconds),
                        "reloaded": self._passages_freed,
                        "idle_minutes": self._idle_seconds / 60,
                    }
                self._passages_freed = False
        if status == "error":
            raise value
        return status, value

    def _iterator_done(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._open_iterators -= 1
                self._last_iterator_end = _clock()

    # --- forwarding ----------------------------------------------------------

    def _call(self, name: str, *args, **kwargs):
        passages = _touches_passages(name, args, kwargs)
        if passages and name != "search":
            # Only where passages have their own ChromaDB collection. Until an
            # index is migrated they sit in the article collection, which
            # every search loads, so every call keeps the worker as before.
            # Asked each time, not cached here: a migration in another
            # process changes the answer, and each search's index_layout()
            # call makes the worker read the layout again.
            try:
                passages = bool(self._request(("call", "separate_passages", (), {}))[1])
            except Exception as e:
                # The answer reads SQLite, which can be locked or damaged. The
                # write must still run (reset_collection is a recovery step);
                # as ordinary activity it keeps the worker, the safe side.
                if not self._layout_check_warned:
                    self._layout_check_warned = True
                    logger.warning(
                        "Could not tell whether passages have their own ChromaDB collection "
                        "(%s); passage writes will keep the semantic search worker like any call",
                        e,
                    )
                passages = False
        return self._request(("call", name, args, kwargs), passages=passages)[1]

    @property
    def embedding_function(self) -> _EmbeddingFunctionProxy:
        proxy = self.__dict__.get("_embedding_function")
        if proxy is None:
            proxy = self.__dict__["_embedding_function"] = _EmbeddingFunctionProxy(self)
        return proxy

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        status, value = self._request(("getattr", name))
        if status == "callable":
            value = functools.partial(self._call, name)
        self.__dict__[name] = value
        return value


class _RemoteIterator:
    """An iterator living in the worker, fetched one item per request.

    While it is open the worker does not exit, which would lose it. It counts
    as closed once exhausted, closed, or dropped.
    """

    def __init__(self, client: WorkerChromaClient, iterator_id: int, generation: int):
        self._client = client
        self._id = iterator_id
        self._generation = generation
        self._open = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self._open:
            raise StopIteration
        try:
            status, value = self._client._request(("next", self._id))
        except BaseException:
            self.close()
            raise
        if status == "done":
            self.close()
            raise StopIteration
        return value

    def close(self) -> None:
        if self._open:
            self._open = False
            self._client._iterator_done(self._generation)

    def __del__(self):
        with contextlib.suppress(Exception):  # at interpreter exit, say
            self.close()


# --- which processes use a worker------------------------------------------------

# Only the long-running server opts in: a CLI command exits when it is done,
# which frees the index anyway, and the worker itself must open the real client.
_enabled = False
# config_path -> ((config mtime, idle minutes), client)
_shared: dict[str | None, tuple[tuple[float, float], WorkerChromaClient]] = {}
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
    try:
        mtime = os.stat(config_path).st_mtime if config_path else 0.0
    except OSError:
        mtime = 0.0
    stamp = (mtime, float(idle_minutes))
    with _shared_lock:
        entry = _shared.get(config_path)
        if entry is not None and entry[0] == stamp:
            return entry[1]
        # A changed config (a new embedding model, say) gets a fresh worker,
        # as it got a fresh in-process client on the next call before.
        stale = entry[1] if entry is not None else None
        client = WorkerChromaClient(config_path, idle_seconds=idle_minutes * 60)
        _shared[config_path] = (stamp, client)
    if stale is not None:
        stale.stop()
    return client


def stop_all() -> None:
    """Stop every worker this process started (server shutdown)."""
    with _shared_lock:
        clients = [client for _stamp, client in _shared.values()]
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
            elif kind == "efattr":
                reply = ("ok", getattr(client.embedding_function, name))
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
                # Some exceptions (the OpenAI SDK's, say) pickle but cannot be
                # unpickled; the caller would get a TypeError, not the error.
                pickle.loads(pickle.dumps(e))
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
