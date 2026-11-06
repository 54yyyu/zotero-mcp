"""Strict SQLite mode: a read that would reach the Zotero API is an error.

In local mode, reads should come from ``zotero.sqlite`` through
:class:`zotero_mcp.library.LibraryBackend`, with the API as a fallback. That
fallback hides any code path that misses the port for as long as Zotero is
running, so developers and people posting patches can turn this mode on to make
such a path fail where it can be seen and fixed. It is a developer check, so it
has no flag or config key: set ``ZOTERO_STRICT_SQLITE=1`` in the environment the
process starts with (a shell, ``.env``, or an MCP client's ``env`` block).

The guard sits at the HTTP layer because that is the one place every pyzotero
call passes through, including clients passed as parameters and method
references that a search of the call sites would miss. It is installed only
when the mode is on, so nobody else runs any of this on their requests.

:class:`StrictSqliteViolation` derives from ``BaseException`` on purpose. Many
read paths catch every ``Exception``, and the existing-item check reads a failed
search as "no match" and goes on to create the item. An ordinary exception would
become an empty result or a false "no match" there, which is the failure this
mode exists to expose. The tool boundary and the CLIs turn it back into a normal
error (``zotero_mcp._context.sync_context``, which wraps every tool and
resource).

Nothing here imports a third-party package at import time: this module is loaded
on every start, and the HTTP libraries are only needed once the mode is on.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
import sys
from typing import Any
from urllib.parse import urlsplit

ENV_VAR = "ZOTERO_STRICT_SQLITE"

_EXEMPT_ATTR = "_zotero_mcp_strict_sqlite_exempt"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Zotero, and Juris-M's port as better_bibtex_client documents it.
_LOCAL_PORTS = {23119, 24119}
_WEB_API_HOST = "api.zotero.org"
_LIBRARY_PATH = re.compile(r"/(users|groups)/")
_READ_METHODS = {"GET", "HEAD"}
# Skipped when naming the caller: they only pass the request along.
_PASS_THROUGH_MODULES = {__name__, "zotero_mcp.library"}

# (owner, attribute, original) for every method the guard replaced. Non-empty
# means the mode is on, so asking whether it is on never touches the disk.
_patches: list[tuple[Any, str, Any]] = []


class StrictSqliteViolation(BaseException):
    """A read would have left SQLite while strict SQLite mode is on."""


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def requested() -> bool:
    """Whether the environment asks for the mode. The one place it is read."""
    return _is_truthy(os.environ.get(ENV_VAR, ""))


def is_active() -> bool:
    return bool(_patches)


def activate_if_requested() -> None:
    """Install the guard if the settings ask for it. Safe to call repeatedly."""
    if not is_active() and requested():
        activate()


def _replace(owner: Any, attribute: str, replacement: Any) -> None:
    _patches.append((owner, attribute, getattr(owner, attribute)))
    setattr(owner, attribute, replacement)


def _guarded_send(original):
    @functools.wraps(original)
    def send(self, request, *args, **kwargs):
        if getattr(self, _EXEMPT_ATTR, None) is None and _is_library_read(request.method, str(request.url)):
            raise StrictSqliteViolation(_refusal(request.method, str(request.url)))
        return original(self, request, *args, **kwargs)

    return send


def activate() -> None:
    """Install the guard. Safe to call repeatedly."""
    if is_active():
        return
    # pyzotero 1.15 moved from httpx to httpx2, and our own code uses both
    # those and requests (Better BibTeX), so guard whichever are installed.
    for module_name, class_name in (("httpx", "Client"), ("httpx2", "Client"), ("requests", "Session")):
        try:
            owner = getattr(importlib.import_module(module_name), class_name)
        except ImportError:
            continue
        _replace(owner, "send", _guarded_send(owner.send))
    sys.stderr.write(
        "Strict SQLite mode is on: reads that would reach the Zotero API are errors. "
        f"Unset {ENV_VAR} to turn it off.\n"
    )


def deactivate() -> None:
    """Remove the guard, restoring the original methods."""
    while _patches:
        owner, attribute, original = _patches.pop()
        setattr(owner, attribute, original)


def exempt(zot: Any, reason: str) -> Any:
    """Allow reads through this pyzotero client, and return it.

    Only for reads SQLite cannot answer by nature. Every use is a place to look
    when asking what the mode does not cover.
    """
    if is_active():
        setattr(zot.client, _EXEMPT_ATTR, reason)
    return zot


def _web_api_paths(host: str, path: str) -> str | None:
    """The path below the web API's root, or None if this isn't the web API."""
    if host == _WEB_API_HOST:
        return path
    configured = os.environ.get("ZOTERO_API_URL", "").strip()
    if configured:
        base = urlsplit(configured)
        if host == base.hostname:
            return path.removeprefix(base.path.rstrip("/"))
    return None


def _is_library_read(method: str, url: str) -> bool:
    parts = urlsplit(url)
    host, path = parts.hostname or "", parts.path
    if host in _LOOPBACK_HOSTS:
        if parts.port not in _local_ports():
            return False
        # Better BibTeX's JSON-RPC reads are POSTs.
        if path.startswith("/better-bibtex/"):
            return True
        if not path.startswith("/api/"):
            return False
        api_path = path.removeprefix("/api")
    else:
        api_path = _web_api_paths(host, path)
        if api_path is None:
            return False
    return method.upper() in _READ_METHODS and _LIBRARY_PATH.match(api_path) is not None


def _local_ports() -> set[int]:
    configured = os.environ.get("ZOTERO_LOCAL_PORT", "").strip()
    return _LOCAL_PORTS | ({int(configured)} if configured.isdigit() else set())


def _call_chain(limit: int = 3) -> str:
    """Our innermost frames, so the message names the code to change."""
    found = []
    frame = sys._getframe(1)
    while frame is not None and len(found) < limit:
        module = frame.f_globals.get("__name__", "")
        if module.startswith("zotero_mcp.") and module not in _PASS_THROUGH_MODULES:
            path = frame.f_code.co_filename.replace(os.sep, "/")
            _, in_package, relative = path.rpartition("/zotero_mcp/")
            filename = relative if in_package else path.rpartition("/")[2]
            found.append(f"{filename}:{frame.f_lineno} in {frame.f_code.co_name}")
        frame = frame.f_back
    return " <- ".join(found) or "unknown caller"


_FIX_HINT = (
    "Port this read to zotero_mcp.library.LibraryBackend. "
    "ZOTERO_STRICT_SQLITE in docs/configuration.md lists what is exempt."
)


def _refusal(method: str, url: str) -> str:
    # Path only: the query string can hold the user's search terms.
    return (
        f"Strict SQLite mode refused {method.upper()} {urlsplit(url).path}. "
        f"Called from: {_call_chain()}. {_FIX_HINT}"
    )


def declined(operation: str, reason: Any) -> StrictSqliteViolation:
    """The error for a read ``SqliteBackend`` declined instead of the API retry."""
    return StrictSqliteViolation(
        f"Strict SQLite mode refused the API fallback for {operation}: {reason}. "
        f"Called from: {_call_chain()}. Add this case to the SQLite backend."
    )


def no_database() -> StrictSqliteViolation:
    """The error for having no local database to read."""
    return StrictSqliteViolation(
        "Strict SQLite mode needs a readable zotero.sqlite in local mode "
        f"(ZOTERO_LOCAL=true), and none is available. Called from: {_call_chain()}."
    )
