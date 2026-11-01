"""The MCP context as the tools use it: from plain synchronous code.

Every tool here is a sync function, which FastMCP runs in a worker thread,
while ``fastmcp.Context.info`` and friends are coroutines. Called bare, they
build a coroutine nobody awaits and the client never sees the message.
:class:`SyncContext` hands each call back to the event loop instead.
"""

import functools

from zotero_mcp import _serverless

if _serverless.ENABLED:
    # zotero-cli: no server in this process, so no FastMCP to import.
    from zotero_mcp._serverless import Context, ToolError
else:
    try:
        from fastmcp import Context
        from fastmcp.exceptions import ToolError
    except ImportError:
        from zotero_mcp._serverless import Context, ToolError


class SyncContext:
    """A ``fastmcp.Context`` whose log methods can be called synchronously."""

    def __init__(self, ctx):
        self._ctx = ctx

    def _log(self, level: str, message: str) -> None:
        import anyio.from_thread

        try:
            anyio.from_thread.run(getattr(self._ctx, level), message)
        except Exception:
            # A progress note must never fail the tool: no event loop to
            # reach (called outside a worker thread) or a client gone away.
            pass

    def info(self, message: str) -> None:
        self._log("info", message)

    def warning(self, message: str) -> None:
        self._log("warning", message)

    def error(self, message: str) -> None:
        self._log("error", message)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _http_request():
    """The HTTP request this tool call arrived on, or None (stdio, in-memory)."""
    try:
        from fastmcp.server.dependencies import get_http_request

        return get_http_request()
    except Exception:
        return None


def http_session_key(request=None) -> str | None:
    """The client session an HTTP tool call belongs to, or None.

    Streamable HTTP carries it in the ``mcp-session-id`` header on
    handshake-era connections (protocol 2025-11-25 and earlier), SSE in the
    ``session_id`` query parameter of each message. Stateless 2026-07-28
    requests carry neither: each is its own connection, so there is no
    session to key anything by (#739).
    """
    request = request if request is not None else _http_request()
    if request is None:
        return None
    session = request.headers.get("mcp-session-id")
    if session:
        return f"http:{session}"
    session = request.query_params.get("session_id")
    if session:
        return f"sse:{session}"
    return None


def on_sessionless_http() -> bool:
    """Whether this tool call came over HTTP with no client session."""
    request = _http_request()
    return request is not None and http_session_key(request) is None


def sync_context(fn):
    """Wrap a sync tool so the ``ctx`` FastMCP injects is a SyncContext.

    Anything else passed as ``ctx`` (the CLI's and the tests' stand-ins) goes
    through untouched. A call from an HTTP client session also runs with that
    session's active library (#739), so one client's zotero_switch_library
    does not move another's; stdio has one client per process and keeps the
    process-wide library. Resources are wrapped too, for the library part.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = kwargs.get("ctx")
        if isinstance(ctx, Context):
            kwargs["ctx"] = SyncContext(ctx)
        session = http_session_key()
        if session is None:
            return fn(*args, **kwargs)
        from zotero_mcp import client as _client

        with _client.library_session(session):
            return fn(*args, **kwargs)

    return wrapper
