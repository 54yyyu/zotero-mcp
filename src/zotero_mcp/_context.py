"""The MCP context as the tools use it: from plain synchronous code.

Every tool here is a sync function, which FastMCP runs in a worker thread,
while ``fastmcp.Context.info`` and friends are coroutines. Called bare, they
build a coroutine nobody awaits and the client never sees the message.
:class:`SyncContext` hands each call back to the event loop instead.
"""

import functools

try:
    from fastmcp import Context
except ImportError:
    class Context:  # type: ignore[no-redef]
        def info(self, message: str) -> None: pass
        def warning(self, message: str) -> None: pass
        def error(self, message: str) -> None: pass


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


def sync_context(fn):
    """Wrap a sync tool so the ``ctx`` FastMCP injects is a SyncContext.

    Anything else passed as ``ctx`` (the CLI's and the tests' stand-ins) goes
    through untouched.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = kwargs.get("ctx")
        if isinstance(ctx, Context):
            kwargs["ctx"] = SyncContext(ctx)
        return fn(*args, **kwargs)

    return wrapper
