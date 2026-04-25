"""Provides fastmcp.Context, falling back to a lightweight stub when fastmcp is not installed."""
try:
    from fastmcp import Context
except ImportError:
    class Context:  # type: ignore[no-redef]
        def info(self, message: str) -> None: pass
        def warning(self, message: str) -> None: pass
        def error(self, message: str) -> None: pass
