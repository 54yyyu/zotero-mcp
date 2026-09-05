"""
Zotero client wrapper for MCP server.
"""

import functools
import json
import logging
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pyzotero import zotero

from zotero_mcp import schema
from zotero_mcp.extract import (
    categorize_attachment,
    extract_file,
    normalize_attachment_priority,
    pick_by_priority,
)
from zotero_mcp.utils import (
    _paginate,
    format_creators,
    html_to_text,
    is_local_mode,
    item_display_date,
    item_display_title,
)
from zotero_mcp.webdav import (
    WebDAVNotConfiguredError,
    download_attachment_from_webdav,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Serialize all Zotero API access. The local API (port 23119) is single-threaded;
# concurrent requests from parallel MCP tool threads queue at the network layer and
# risk hitting pyzotero's 30s timeout. A process-local lock ensures only one
# request is in-flight at a time — the rest queue in-process (microseconds) instead
# of at the API (seconds/timeout). RLock allows nested calls from the same thread.
_zotero_api_lock = threading.RLock()

# Bound how long a tool will WAIT to acquire the lock before giving up. Without a
# bound, a single slow/stuck op (e.g. a hung cloud write or PDF upload) holds the
# lock and every other tool — reads included — blocks behind it until FastMCP's
# ~60s client timeout fires, surfacing as an opaque "-32001 Request timed out" on
# every queued call. A bounded acquire turns that into a fast, actionable error
# for the *waiters* while leaving the in-flight op untouched. Keep this safely
# below the client timeout. Override via ZOTERO_MCP_LOCK_TIMEOUT (seconds; <=0
# restores the old unbounded behaviour).
_DEFAULT_LOCK_TIMEOUT = 45.0


def _lock_timeout() -> float:
    raw = os.getenv("ZOTERO_MCP_LOCK_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_LOCK_TIMEOUT
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_LOCK_TIMEOUT


class ZoteroApiBusyError(RuntimeError):
    """Raised when the per-process Zotero API lock can't be acquired in time.

    Signals that another Zotero operation is still in flight (likely slow or
    stuck) — not that this call itself failed. Callers should surface a clear,
    retryable message rather than letting the request hang to a timeout.
    """


@contextmanager
def zotero_api_lock():
    """Hold the shared Zotero API lock for a block of code.

    The context-manager form of :func:`with_zotero_api_lock`, with identical
    semantics (bounded acquire, ZoteroApiBusyError for waiters, RLock
    reentrancy, ZOTERO_MCP_LOCK_TIMEOUT honoured). Use it when only part of a
    tool touches the Zotero API and the rest is long local work — parsing a
    downloaded PDF, running a subprocess — that no other tool needs to be
    blocked behind (#431).
    """
    timeout = _lock_timeout()
    if timeout <= 0:
        # Opt-out: original unbounded behaviour.
        with _zotero_api_lock:
            yield
        return
    acquired = _zotero_api_lock.acquire(timeout=timeout)
    if not acquired:
        raise ZoteroApiBusyError(
            f"Another Zotero API operation is still in progress and did not "
            f"release within {timeout:.0f}s. This usually means a previous "
            f"call is slow or stuck (e.g. a large PDF upload or an "
            f"unreachable Zotero cloud). Please retry shortly; if it "
            f"persists, restart the Zotero MCP server."
        )
    try:
        yield
    finally:
        _zotero_api_lock.release()


def with_zotero_api_lock(func):
    """Serialize Zotero API access across concurrent MCP tool threads.

    Acquires the shared RLock with a bounded wait so a stuck op can't wedge
    every other tool into an opaque client timeout. The lock is reentrant, so
    nested decorated calls on the same thread (e.g. add_by_url -> add_by_doi)
    acquire instantly and are never blocked by this bound.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with zotero_api_lock():
            return func(*args, **kwargs)
    return wrapper


# Runtime library override state — set by zotero_switch_library tool.
# When non-empty, these values override the corresponding environment variables
# in get_zotero_client(). Keys: "library_id", "library_type".
_active_library_override: dict[str, str] = {}


def set_active_library(library_id: str, library_type: str) -> None:
    """Set runtime library override for all subsequent get_zotero_client() calls."""
    _active_library_override["library_id"] = library_id
    _active_library_override["library_type"] = library_type


def clear_active_library() -> None:
    """Clear runtime library override, reverting to environment variable defaults."""
    _active_library_override.clear()


def get_active_library() -> dict[str, str]:
    """Return the current active library override (empty dict if using defaults)."""
    return dict(_active_library_override)


def get_active_group_id() -> int:
    """group_id (0 = personal, else Zotero groupID) of the library
    ``get_zotero_client()`` is currently scoped to."""

    override = _active_library_override
    library_id = override.get("library_id") or os.getenv("ZOTERO_LIBRARY_ID") or "0"
    library_type = override.get("library_type") or os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    if library_type == "group":
        try:
            return int(library_id)
        except (TypeError, ValueError):
            return 0
    return 0


# Timeouts for the injected local httpx client. pyzotero only passes an
# explicit per-request timeout on its READ paths; Zotero._write() and
# authorize_local() hand the request straight to self.client with no timeout
# kwarg. pyzotero's own default client sets one at construction, but we supply
# our own client (for the HTTP/1.1 pin below), so without these constants every
# local write would inherit httpx's 5s default — and authorize_local() would
# time out about five seconds into a modal dialog nobody could answer that fast.
_LOCAL_READ_TIMEOUT = 30.0
_LOCAL_WRITE_TIMEOUT = 120.0
# Kept under FastMCP's ~60s client timeout so the MCP tool reports a real
# outcome instead of the transport dying first.
_LOCAL_AUTHORIZE_TIMEOUT = 55.0


def _make_local_http_client(timeout: float = _LOCAL_READ_TIMEOUT) -> httpx.Client:
    """Return an httpx.Client pinned to HTTP/1.1 for the local Zotero server.

    Zotero 8's local server (port 23119) only speaks HTTP/1.0. httpx defaults
    to attempting HTTP/2 negotiation, which the local server rejects with 502
    Bad Gateway — every tool call fails even though the MCP starts cleanly
    (#160). Forcing http1=True / http2=False on the transport keeps requests
    on HTTP/1.1 and the local API answers normally.

    The explicit timeout is load-bearing, not tidying: see the comment on the
    constants above. Connect stays short because the server is on loopback —
    if it isn't listening we want to know immediately, not in two minutes.
    """
    return httpx.Client(
        transport=httpx.HTTPTransport(http1=True, http2=False),
        follow_redirects=True,
        timeout=httpx.Timeout(timeout, connect=5.0),
    )


def _apply_default_headers(zot: zotero.Zotero) -> zotero.Zotero:
    """Restore the pyzotero headers that supplying our own client drops.

    pyzotero only calls default_headers() when it builds the httpx client
    itself, so an injected client sends neither User-Agent nor
    Zotero-API-Version: 3. Harmless while we only read; worth having once we
    start writing.

    Authorization is stripped deliberately. In hybrid mode this client is
    built with the zotero.org API key, and default_headers() turns that into
    `Authorization: Bearer <key>` — which would send a cloud credential to a
    local HTTP server that has no use for it. Local writes authenticate with
    Zotero-API-Key instead, attached per-request by pyzotero.
    """
    try:
        headers = {
            name: value
            for name, value in zot.default_headers().items()
            if name.lower() != "authorization"
        }
        zot.client.headers.update(headers)
    except Exception:
        pass
    return zot


@dataclass
class AttachmentDetails:
    """Details about a Zotero attachment."""

    key: str
    title: str
    filename: str
    content_type: str


@dataclass
class AttachmentDownloadResult:
    """Result of downloading an attachment from one of the supported sources."""

    path: Path | None
    source: str | None
    errors: list[str]


def get_zotero_client() -> zotero.Zotero:
    """
    Get authenticated Zotero client using environment variables.

    If a runtime library override is active (via set_active_library()),
    those values take precedence over environment variables.

    Returns:
        A configured Zotero client instance.

    Raises:
        ValueError: If required environment variables are missing.
    """
    # Runtime overrides take precedence over environment variables
    override = _active_library_override
    library_id = override.get("library_id") or os.getenv("ZOTERO_LIBRARY_ID")
    library_type = override.get("library_type") or os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    api_key = os.getenv("ZOTERO_API_KEY")
    local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]

    # For local API, default to user ID 0 if not specified
    if local and not library_id:
        library_id = "0"

    # For remote API, we need both library_id and api_key
    if not local and not (library_id and api_key):
        raise ValueError(
            "Missing required environment variables. Please set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY, "
            "or use ZOTERO_LOCAL=true for local Zotero instance."
        )

    zot = zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        api_key=api_key,
        local=local,
        client=_make_local_http_client() if local else None,
    )
    return _apply_default_headers(zot) if local else zot


def get_local_zotero_client() -> zotero.Zotero | None:
    """
    Get a local Zotero client for file access (WebDAV/local storage).

    This client connects to the local Zotero instance running on port 23119.
    It's useful for accessing PDF files stored via WebDAV when the main
    client is configured for web API.

    Returns:
        A local Zotero client instance, or None if local Zotero is not available.
    """
    try:
        # Create a local client - library_id 0 is the default for local.
        # HTTP/1.1-only transport for compatibility with Zotero 8's local
        # server (#160) — httpx default HTTP/2 negotiation returns 502.
        client = _apply_default_headers(
            zotero.Zotero(
                library_id="0",
                library_type="user",
                api_key=None,
                local=True,
                client=_make_local_http_client(),
            )
        )
        # Test connection by making a simple request
        client.items(limit=1)
        return client
    except Exception:
        return None


def get_web_zotero_client() -> zotero.Zotero | None:
    """
    Get a web API Zotero client for write operations.

    This client connects to the Zotero web API and can create/modify items.
    Requires ZOTERO_API_KEY and ZOTERO_LIBRARY_ID environment variables.

    Returns:
        A web API Zotero client instance, or None if credentials are not available.
    """
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    api_key = os.getenv("ZOTERO_API_KEY")

    if not library_id or not api_key:
        return None

    return zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        api_key=api_key,
        local=False,
    )


def is_local_zotero_available() -> bool:
    """Check if local Zotero instance is running and accessible."""
    client = get_local_zotero_client()
    return client is not None


# ---------------------------------------------------------------------------
# Local API write access (Zotero 10+)
#
# Writing through the local API needs two things the read path never sends: a
# Zotero-Server-ID header identifying the Zotero database, and a local API key
# the user grants through a dialog in the Zotero app. The key is unrelated to
# a zotero.org API key and cannot be created ahead of time.
# ---------------------------------------------------------------------------

ZOTERO_MCP_CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"

LOCAL_API_KEY_ENV = "ZOTERO_LOCAL_API_KEY"
LOCAL_SERVER_ID_ENV = "ZOTERO_LOCAL_SERVER_ID"
LOCAL_WRITE_ENV = "ZOTERO_LOCAL_WRITE"
DEFAULT_APP_NAME = "Zotero MCP"

# Credentials granted during this process, before/instead of anything on disk.
_local_write_state: dict[str, Any] = {}
# Cached capability probe: {"server_id": str | None, "checked_at": float}
_local_probe_cache: dict[str, Any] = {}
# A negative probe is re-checked after this long, so upgrading or restarting
# Zotero mid-session recovers without restarting the MCP server.
_LOCAL_PROBE_NEGATIVE_TTL = 60.0

# Deliberately NOT _zotero_api_lock. Authorization blocks until a human clicks
# a button; holding the shared API lock that long would push every other tool
# past its bounded 45s acquire. This lock exists only to stop two dialogs from
# being opened at once, so it is always acquired non-blocking.
_local_auth_lock = threading.Lock()


class ZoteroAuthInProgressError(RuntimeError):
    """Raised when an authorization dialog is already open."""


def load_zotero_mcp_config() -> dict:
    """Return the parsed ``~/.config/zotero-mcp/config.json``, or ``{}``.

    Missing file or parse errors yield an empty dict so callers can use
    ``.get(...)`` chains without guarding. Callers that intend to write the
    file back must use ``_readable_config_for_update`` instead — an empty dict
    here does not distinguish "no config" from "could not read it", and
    rewriting on the latter would discard the user's other settings.
    """
    try:
        return _readable_config_for_update()
    except OSError:
        return {}


def _readable_config_for_update() -> dict:
    """Parse the config file for a read-modify-write. Raises if unreadable.

    A config that exists but cannot be parsed must not be silently replaced:
    it also holds ``semantic_search`` and ``client_env``, and treating a
    transient read failure or a hand-editing typo as "empty" would drop both.
    """
    if not ZOTERO_MCP_CONFIG_PATH.exists():
        return {}
    try:
        with open(ZOTERO_MCP_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f) or {}
    except json.JSONDecodeError as e:
        raise OSError(
            f"{ZOTERO_MCP_CONFIG_PATH} is not valid JSON ({e}). Fix or remove "
            "the file; refusing to overwrite it and lose the other settings."
        ) from e


def _local_write_enabled() -> bool:
    """Honour the ZOTERO_LOCAL_WRITE kill switch. Unset or "auto" means on."""
    raw = os.getenv(LOCAL_WRITE_ENV, "").strip().lower()
    if not raw or raw == "auto":
        return True
    return raw in {"true", "yes", "1"}


def probe_local_server_id(timeout: float = 3.0, force: bool = False) -> str | None:
    """Return the local Zotero server ID, or None if this build has no writes.

    Every local API response carries a Zotero-Server-ID header; a build that
    predates local write support carries none. GET /api/ (the trailing slash
    matters — bare /api is a 404) is the cheapest way to ask.

    Never called at import or from the write path itself: only the capability
    tool, the CLI, and the message shown when writes are unavailable.
    """
    cached_at = _local_probe_cache.get("checked_at")
    if not force and cached_at is not None:
        server_id = _local_probe_cache.get("server_id")
        if server_id or (time.monotonic() - cached_at) < _LOCAL_PROBE_NEGATIVE_TTL:
            return server_id

    # Deliberately not ZOTERO_LOCAL_PORT: pyzotero hardcodes localhost:23119
    # for local mode, so probing anywhere else could report "writes supported"
    # for a server the writes will never reach.
    server_id = None
    try:
        with _make_local_http_client(timeout) as http:
            resp = http.get("http://localhost:23119/api/")
            server_id = resp.headers.get("zotero-server-id")
    except Exception:
        server_id = None

    _local_probe_cache["server_id"] = server_id
    _local_probe_cache["checked_at"] = time.monotonic()
    return server_id


def get_local_write_credentials() -> tuple[str | None, str | None, str | None]:
    """Return (api_key, server_id, source) for local writes.

    Precedence: credentials granted in this process, then the environment,
    then the config file. Source is "session", "env", "config" or None.
    """
    if key := _local_write_state.get("key"):
        return key, _local_write_state.get("server_id"), "session"

    if key := os.getenv(LOCAL_API_KEY_ENV, "").strip():
        return key, os.getenv(LOCAL_SERVER_ID_ENV, "").strip() or None, "env"

    stored = load_zotero_mcp_config().get("local_api") or {}
    if key := stored.get("key"):
        return key, stored.get("server_id"), "config"

    return None, None, None


def _local_key_remembered() -> bool | None:
    """Whether the active key was granted with "Always Allow", if we know."""
    _key, _server_id, source = get_local_write_credentials()
    if source == "session":
        return _local_write_state.get("remember")
    if source == "config":
        return (load_zotero_mcp_config().get("local_api") or {}).get("remember")
    return None


def _write_config(config: dict) -> None:
    """Persist the config file, owner-only, replacing it atomically."""
    ZOTERO_MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ZOTERO_MCP_CONFIG_PATH.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    # The file holds a credential — keep it owner-only. Best-effort; a no-op
    # on platforms without POSIX permissions. Set before the rename so the
    # key is never briefly world-readable at its final name.
    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass
    os.replace(temp_path, ZOTERO_MCP_CONFIG_PATH)


def store_local_write_credentials(
    key: str, server_id: str | None, remember: bool, app_name: str = DEFAULT_APP_NAME
) -> Path | None:
    """Cache credentials for this process and persist them. Returns the path.

    Persisted under its own ``local_api`` section rather than ``client_env``,
    which ``setup_helper._write_standalone_config`` rebuilds from scratch on
    every ``zotero-mcp setup`` run and would silently drop the key.

    Returns None if the file could not be written, including when it exists
    but cannot be parsed — the session copy is still set, so the caller can
    keep working and warn.
    """
    _local_write_state.update({"key": key, "server_id": server_id, "remember": remember})

    try:
        config = _readable_config_for_update()
        config["local_api"] = {
            "key": key,
            "server_id": server_id,
            "remember": remember,
            "app_name": app_name,
            "granted": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _write_config(config)
        return ZOTERO_MCP_CONFIG_PATH
    except OSError:
        return None


def clear_local_write_credentials() -> bool:
    """Forget the local API key. True if anything was actually cleared.

    Covers the session copy as well as the stored one, so a key granted during
    this run reports honestly instead of "nothing to remove". A key supplied
    through the environment cannot be cleared from here — the caller is
    expected to say so.
    """
    cleared = bool(_local_write_state)
    _local_write_state.clear()

    try:
        config = _readable_config_for_update()
    except OSError:
        return cleared
    if "local_api" not in config:
        return cleared
    del config["local_api"]
    try:
        _write_config(config)
        return True
    except OSError:
        return cleared


def get_local_write_client() -> zotero.Zotero | None:
    """Return a local Zotero client authorized to write, or None.

    None — never an exception — when local writes are switched off, when we
    aren't in local mode, or when no key has been granted yet. Callers treat
    that as "fall back to the web API".
    """
    if not _local_write_enabled() or not is_local_mode():
        return None

    key, server_id, _source = get_local_write_credentials()
    if not key:
        return None

    override = _active_library_override
    library_type = override.get("library_type") or os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    library_id = override.get("library_id") or os.getenv("ZOTERO_LIBRARY_ID")
    # The local API addresses the user library as users/0 whatever the web
    # user id is; a group keeps its real numeric id.
    if not library_type.startswith("group"):
        library_id = "0"
    elif not library_id:
        # A group with no id is a broken configuration, but this factory
        # promises None rather than an exception — pyzotero would raise
        # MissingCredentialsError, which no caller is catching.
        return None

    zot = zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        api_key=None,
        local=True,
        client=_make_local_http_client(_LOCAL_WRITE_TIMEOUT),
        # Supplying the cached id spares pyzotero a GET /api/ before the first
        # write of every tool call, since each call builds a fresh client.
        server_id=server_id or None,
        local_api_key=key,
    )
    return _apply_default_headers(zot)


def authorize_local_api(
    app_name: str = DEFAULT_APP_NAME,
    timeout: float = _LOCAL_AUTHORIZE_TIMEOUT,
    persist: bool = True,
) -> dict:
    """Ask Zotero for a local API key. Blocks on a dialog in the Zotero app.

    Returns pyzotero's response — a ``key`` and a ``remember`` flag — with the
    server id added. ``remember`` is False when the user chose "Allow" rather
    than "Always Allow", which makes the key valid for exactly one write.

    Raises ZoteroAuthInProgressError if a dialog is already open, and lets
    pyzotero's LocalAPIDeniedError / TooManyRequestsError / UnsupportedParams-
    Error through for the caller to phrase.
    """
    if not _local_auth_lock.acquire(blocking=False):
        raise ZoteroAuthInProgressError(
            "A Zotero authorization dialog is already open. Answer it in "
            "Zotero, then retry."
        )
    try:
        zot = zotero.Zotero(
            library_id="0",
            library_type="user",
            api_key=None,
            local=True,
            client=_make_local_http_client(timeout),
        )
        result = dict(zot.authorize_local(app_name))
        result["server_id"] = zot.server_id
        remember = bool(result.get("remember"))

        # A single-use key is never written to disk. It is consumed by the
        # first write, and a dead key on disk outranks working web credentials
        # on every later run — which would turn one stray click on "Allow"
        # into a permanently broken hybrid setup. Keeping it in the session
        # still lets this process use it once, and still lets the 401 handler
        # explain what happened.
        if persist and remember:
            result["stored_at"] = store_local_write_credentials(
                result["key"], zot.server_id, remember, app_name
            )
        else:
            _local_write_state.update(
                {
                    "key": result["key"],
                    "server_id": zot.server_id,
                    "remember": remember,
                }
            )
        return result
    finally:
        _local_auth_lock.release()


def get_write_capabilities(probe: bool = True) -> dict[str, Any]:
    """Describe how (and whether) writes can reach Zotero right now."""
    local_mode = is_local_mode()
    key, server_id, source = get_local_write_credentials()
    has_web = bool(os.getenv("ZOTERO_LIBRARY_ID") and os.getenv("ZOTERO_API_KEY"))
    probed_id = probe_local_server_id() if (probe and local_mode) else None

    if not local_mode:
        mode = "web" if has_web else "none"
    elif key and _local_write_enabled():
        mode = "local"
    elif has_web:
        mode = "hybrid"
    else:
        mode = "none"

    return {
        "mode": mode,
        "local_mode": local_mode,
        "local_write_enabled": _local_write_enabled(),
        "server_supports_local_writes": bool(probed_id),
        "server_id": probed_id or server_id,
        "has_local_key": bool(key),
        "local_key_source": source,
        "local_key_remember": _local_key_remembered(),
        "has_web_credentials": has_web,
    }


def format_item_metadata(item: dict[str, Any], include_abstract: bool = True) -> str:
    """
    Format a Zotero item's metadata as markdown.

    Args:
        item: A Zotero item dictionary.
        include_abstract: Whether to include the abstract in the output.

    Returns:
        Markdown-formatted metadata.
    """
    data = item.get("data", {})
    item_type = data.get("itemType", "unknown")

    # Type-specific base fields (a statute's title is "nameOfAct"), notes
    # (whose title is their first line) and standalone attachments (which have
    # only a filename) all resolve here, so a search result and an item lookup
    # can never disagree about what something is called (#452, #447).
    heading = item_display_title(data)

    # Basic information
    lines = [
        f"# {heading}",
        f"**Type:** {item_type}",
        f"**Item Key:** {data.get('key')}",
    ]

    # Trash status. The Zotero web API returns data.deleted=1 for items in
    # the Trash; prior versions silently rendered trashed items as if live,
    # so agents reasoning about "current" state could cite papers the user
    # had explicitly removed. Surface it near the top where it's hard to miss.
    if data.get("deleted"):
        lines.append("**Status:** 🗑️ In Trash (recoverable from Zotero Trash view)")

    # Date. Resolved the same way as the title: a case's date is
    # `dateDecided`, a statute's `dateEnacted`, a patent's `issueDate` (#452).
    if date := item_display_date(data):
        lines.append(f"**Date:** {date}")

    # Authors/Creators
    if creators := data.get("creators", []):
        lines.append(f"**Authors:** {format_creators(creators)}")

    # Publication details based on item type
    if item_type == "journalArticle":
        if journal := data.get("publicationTitle"):
            journal_info = f"**Journal:** {journal}"
            if volume := data.get("volume"):
                journal_info += f", Volume {volume}"
            if issue := data.get("issue"):
                journal_info += f", Issue {issue}"
            if pages := data.get("pages"):
                journal_info += f", Pages {pages}"
            lines.append(journal_info)
    elif item_type == "bookSection":
        if book_title := data.get("bookTitle"):
            lines.append(f"**Book:** {book_title}")
        if pages := data.get("pages"):
            lines.append(f"**Pages:** {pages}")

    # Publisher and place — emitted as independent labeled lines for any
    # item type that has them (book, bookSection, thesis, report, etc.).
    # Round-trip parity: agents that read these need a stable, labeled form.
    if publisher := data.get("publisher"):
        lines.append(f"**Publisher:** {publisher}")
    if place := data.get("place"):
        lines.append(f"**Place:** {place}")

    # Identifiers and URL
    if doi := data.get("DOI"):
        lines.append(f"**DOI:** {doi}")
    if isbn := data.get("ISBN"):
        lines.append(f"**ISBN:** {isbn}")
    if issn := data.get("ISSN"):
        lines.append(f"**ISSN:** {issn}")
    if url := data.get("url"):
        lines.append(f"**URL:** {url}")

    # Whatever the branches above did not cover. The formatter knows about
    # journalArticle and bookSection by name and renders nothing type-specific
    # for anything else, so a case lost `court`, `docketNumber` and `reporter`
    # entirely — the whole reason its record looked empty (#452). Rather than
    # adding a branch per type forever, ask the schema what fields this type
    # has and show the populated ones that are not already above.
    _shown = {
        "key", "itemType", "title", "date", "creators", "tags", "collections",
        "relations", "abstractNote", "extra", "note", "deleted", "version",
        "dateAdded", "dateModified", "parentItem", "filename", "contentType",
        "publicationTitle", "bookTitle", "volume", "issue", "pages",
        "publisher", "place", "DOI", "ISBN", "ISSN", "url", "shortTitle",
        "accessDate", "libraryCatalog", "callNumber", "archive",
        "archiveLocation", "rights", "language",
    }
    # The type's own spellings of title and date are already in the heading
    # and the Date line; showing them again under their raw names would read
    # as two different pieces of information.
    try:
        _shown.add(schema.resolve_field(item_type, "title"))
        _shown.add(schema.resolve_field(item_type, "date"))
        _type_fields = schema.valid_fields(item_type)
    except Exception:
        _type_fields = set()

    _extra_lines = []
    for field in sorted(_type_fields - _shown):
        value = data.get(field)
        if value and isinstance(value, str):
            # camelCase -> "Docket Number"
            label = re.sub(r"(?<!^)(?=[A-Z])", " ", field).title()
            _extra_lines.append(f"**{label}:** {value}")
    lines.extend(_extra_lines)

    # Extra field often holds citation key / misc metadata
    if extra := data.get("extra"):
        lines.extend(["", "## Extra", extra])

        # Try to surface a citation key if present in Extra
        for line in extra.splitlines():
            if "citation key" in line.lower():
                key_part = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                lines.append(f"**Citation Key (from Extra):** {key_part}")
                break

    # Tags
    if tags := data.get("tags"):
        tag_list = [f"`{tag['tag']}`" for tag in tags]
        if tag_list:
            lines.append(f"**Tags:** {' '.join(tag_list)}")

    # Abstract
    if include_abstract and (abstract := data.get("abstractNote")):
        lines.extend(["", "## Abstract", abstract])

    # A note's body IS its content -- there is nothing else to show, and
    # returning the metadata alone told a caller the note existed while
    # withholding the only thing they asked for (#447).
    if item_type == "note" and (note_body := data.get("note")):
        lines.extend(["", "## Note", html_to_text(note_body)])

    # Related Items (dc:relation URIs → item keys)
    dc_relations = data.get("relations", {}).get("dc:relation", [])
    if isinstance(dc_relations, str):
        dc_relations = [dc_relations]
    if dc_relations:
        related_keys = [uri.rstrip("/").split("/")[-1] for uri in dc_relations]
        lines.extend(["", "## Related Items", *[f"- {k}" for k in related_keys]])

    # Collections — list actual keys rather than a bare count. The Zotero
    # web API does NOT cascade collection-delete to items, so the array
    # can contain dangling references to collections that no longer exist.
    # Showing the keys lets agents verify against zotero_search_collections
    # instead of trusting a potentially stale count.
    if collections := data.get("collections", []):
        lines.append(f"**Collections:** {', '.join(collections)}")

    # Notes - this requires additional API calls, so we just indicate if there are notes
    if "meta" in item and item["meta"].get("numChildren", 0) > 0:
        lines.append(f"**Notes/Attachments:** {item['meta']['numChildren']}")

    return "\n\n".join(lines)


def generate_bibtex(item: dict[str, Any]) -> str:
    """
    Generate BibTeX format for a Zotero item.

    Args:
        item: Zotero item data

    Returns:
        BibTeX formatted string
    """
    data = item.get("data", {})
    # The key lives at ``data.key`` in Zotero API responses but only at the
    # top level in some locally-assembled items; accept either.
    item_key = data.get("key") or item.get("key")

    # A trashed item exports a perfectly plausible entry, and BibTeX has no
    # field that would say otherwise — markdown shows a trash status and JSON
    # carries data.deleted, but a BibTeX consumer sees nothing. That gap is
    # what made a trashed duplicate look like a healthy record to a caller
    # reading only this format. A comment line is inert to every BibTeX
    # parser and visible to every human.
    trash_marker = "% Status: in trash\n" if data.get("deleted") else ""

    # Try Better BibTeX first — it produces better entries and the user's
    # real pinned citekeys. Any failure falls through to the local generator
    # below; BBT is an enhancement, never a prerequisite.
    try:
        from zotero_mcp.better_bibtex_client import ZoteroBetterBibTexAPI

        if item_key:
            bibtex = ZoteroBetterBibTexAPI()

            if bibtex.is_zotero_running():
                exported = bibtex.export_bibtex(item_key)
                # Guard the result even though export_bibtex now raises on a
                # blank export: returning "" here is indistinguishable from
                # "no BibTeX" and from "no such item", which is the whole
                # defect being fixed.
                if exported and exported.strip():
                    return trash_marker + exported
                logger.warning(
                    "Better BibTeX returned no entry for %s; "
                    "falling back to local BibTeX generation", item_key
                )

    except Exception as e:
        # BBT does not index trashed items and returns a null citekey for
        # them, so a perfectly readable item can fail here. Fall back rather
        # than fail: the local generator below works from the item data we
        # already hold.
        logger.warning(
            "Better BibTeX export failed for %s (%s); "
            "falling back to local BibTeX generation", item_key, e
        )

    # Fallback to basic BibTeX generation
    item_type = data.get("itemType", "misc")

    if item_type in ["attachment", "note"]:
        raise ValueError(f"Cannot export BibTeX for item type '{item_type}'")

    # Map Zotero item types to BibTeX types
    type_map = {
        "journalArticle": "article",
        "book": "book",
        "bookSection": "incollection",
        "conferencePaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
        "webpage": "misc",
        "manuscript": "unpublished"
    }

    # Create citation key
    creators = data.get("creators", [])
    author = ""
    if creators:
        first = creators[0]
        author = first.get("lastName", first.get("name", "").split()[-1] if first.get("name") else "").replace(" ", "")

    # Resolved, not read straight off `date` — a case keeps its date in
    # `dateDecided`, a statute in `dateEnacted` (#452). Take the first
    # four-digit run rather than the leading four characters: these are
    # display strings ("8 October 2024"), not ISO dates.
    _display_date = item_display_date(data)
    _year_match = re.search(r"\b(\d{4})\b", _display_date) if _display_date else None
    year = _year_match.group(1) if _year_match else "nodate"
    # ``item_key`` can be absent on items assembled locally; never render the
    # literal string "None" into a citekey.
    cite_key = f"{author}{year}_{item_key}" if item_key else f"{author}{year}"
    if not cite_key:
        cite_key = "untitled"

    # Build BibTeX entry
    bib_type = type_map.get(item_type, "misc")
    lines = [f"@{bib_type}{{{cite_key},"]

    # Add fields. `title` and `date` are resolved through the base-field map
    # first, so a case or statute exports with its name and year rather than
    # as an empty `@misc{nodate_KEY}` (#452).
    resolved = dict(data)
    if title := item_display_title(data):
        if title not in ("Untitled", "Untitled Note"):
            resolved["title"] = title

    # No ("date", "year") entry: the block below already emits `year` from the
    # resolved four-digit year. Mapping it here as well emitted it twice, the
    # first time as the raw display string ("8 October 2024"), which is not a
    # BibTeX year.
    field_mappings = [
        ("title", "title"),
        ("publicationTitle", "journal"),
        ("bookTitle", "booktitle"),
        ("volume", "volume"),
        ("issue", "number"),
        ("pages", "pages"),
        ("publisher", "publisher"),
        ("place", "address"),
        ("DOI", "doi"),
        ("ISBN", "isbn"),
        ("ISSN", "issn"),
        ("url", "url"),
        ("abstractNote", "abstract")
    ]

    for zotero_field, bibtex_field in field_mappings:
        if value := resolved.get(zotero_field):
            # Escape special characters
            value = value.replace("{", "\\{").replace("}", "\\}")
            lines.append(f'  {bibtex_field} = {{{value}}},')

    # Add authors
    if creators:
        authors = []
        for creator in creators:
            if creator.get("creatorType") == "author":
                if "lastName" in creator and "firstName" in creator:
                    authors.append(f"{creator['lastName']}, {creator['firstName']}")
                elif "name" in creator:
                    authors.append(creator["name"])
        if authors:
            lines.append(f'  author = {{{" and ".join(authors)}}},')

    # Add year
    if year != "nodate":
        lines.append(f'  year = {{{year}}},')

    # Remove trailing comma from last field and close entry
    if lines[-1].endswith(','):
        lines[-1] = lines[-1][:-1]
    lines.append("}")

    return trash_marker + "\n".join(lines)


def configured_attachment_priority() -> tuple[str, ...]:
    """The user's ``attachment_priority``, or the default if unset/unreadable.

    Imported lazily because ``config`` is not a dependency of this module at
    import time and a missing config must never break attachment lookup.
    """
    try:
        from zotero_mcp.config import load_config

        configured = load_config().semantic_search.extraction.attachment_priority
    except Exception:
        return normalize_attachment_priority(None)
    return normalize_attachment_priority(configured)


def get_attachment_details(
    zot: zotero.Zotero, item: dict[str, Any], priority=None
) -> AttachmentDetails | None:
    """
    Get attachment details for a Zotero item, finding the most relevant attachment.

    Passing a key that names an attachment *directly* returns that attachment
    unchanged, without consulting ``priority``. That short-circuit is
    supported and deliberate: it is how a caller reads one specific file on
    an item that has several (#378).

    Args:
        zot: A Zotero client instance.
        item: A Zotero item dictionary.
        priority: Attachment kinds in preference order. ``None`` uses the
            configured ``attachment_priority``. Callers that need one
            specific kind should say so — ``zotero_read_pdf_pages`` passes
            ``("pdf",)`` so a markdown-first configuration cannot hand it a
            file it has no way to read.

    Returns:
        AttachmentDetails if found, None otherwise.
    """
    data = item.get("data", {})
    item_type = data.get("itemType")
    # Top-level "key" is the reliable one: some API responses (and the local
    # Zotero server) omit it from the nested data object (#372).
    item_key = data.get("key") or item.get("key")

    # Direct attachment
    if item_type == "attachment":
        return AttachmentDetails(
            key=item_key,
            title=data.get("title", "Untitled"),
            filename=data.get("filename", ""),
            content_type=data.get("contentType", ""),
        )

    if priority is None:
        priority = configured_attachment_priority()

    # For regular items, look for child attachments
    try:
        children = _paginate(zot.children, item_key)

        candidates = []
        for child in children:
            child_data = child.get("data", {})
            if child_data.get("itemType") != "attachment":
                continue
            content_type = child_data.get("contentType", "")
            filename = child_data.get("filename", "")
            title = child_data.get("title", "Untitled")
            key = child.get("key", "")

            # Unlike the local path, an attachment we cannot parse ourselves
            # is still worth returning: the caller tries Zotero's
            # server-side fulltext index first, which covers formats we have
            # no reader for (EPUB, DOCX). So an uncategorized attachment
            # joins the catch-all bucket rather than being dropped.
            category = categorize_attachment(filename or title, content_type) or "other"

            # The API exposes no file size, so this only distinguishes a
            # stored file (32-char md5) from a linked one (no md5) — enough
            # to prefer a real upload, not a true size ordering.
            size_proxy = len(child_data.get("md5") or "")

            candidates.append((
                category,
                size_proxy,
                AttachmentDetails(
                    key=key,
                    title=title,
                    filename=filename,
                    content_type=content_type,
                ),
            ))

        if (chosen := pick_by_priority(candidates, priority)) is not None:
            return chosen
    except Exception:
        pass

    return None


def download_attachment_file(
    attachment_key: str,
    destination_dir: str | Path,
    filename: str | None = None,
    *,
    local_client: zotero.Zotero | None = None,
    web_client: zotero.Zotero | None = None,
    enable_webdav: bool = True,
) -> AttachmentDownloadResult:
    """
    Download an attachment using the best available source.

    The fallback order is:
    1. local Zotero storage, resolved straight off the local SQLite DB
    2. local Zotero API (works with local storage or desktop-managed WebDAV)
    3. Direct WebDAV access via environment variables
    4. Zotero Web API (works with Zotero cloud storage)

    Step 1 exists because none of the later steps can serve a *linked* file.
    The local API answers ``/file`` for a linked attachment with a 302 to a
    ``file://`` URL, which httpx refuses to follow ("unsupported protocol"),
    and a linked file is by definition never uploaded to WebDAV or Zotero
    storage — so all three remaining sources fail on an attachment that is
    sitting readable on the same disk. Reading the path out of the DB avoids
    the redirect entirely and is faster for ordinary stored files too.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target_name = Path(filename or f"{attachment_key}.bin").name
    target_path = destination / target_name
    errors: list[str] = []

    def _cleanup_target() -> None:
        if target_path.exists() and target_path.stat().st_size == 0:
            target_path.unlink()

    def _try_local_storage() -> AttachmentDownloadResult | None:
        """Resolve the attachment off disk via the local Zotero SQLite DB.

        Gated on local mode so a web-API user pointed at a group library never
        matches an unrelated same-key row in a personal DB that happens to
        exist on the machine.
        """
        try:
            from zotero_mcp.config import load_config
            from zotero_mcp.local_db import LocalZoteroReader
            from zotero_mcp.utils import is_local_mode

            if not is_local_mode():
                return None

            with LocalZoteroReader(db_path=load_config().resolve_zotero_db_path()) as reader:
                attachment = reader.get_attachment_by_key(attachment_key)
                if attachment is None:
                    return None

                resolved = reader._resolve_attachment_path(
                    attachment_key, attachment["zotero_path"] or ""
                )
                if not (resolved and resolved.exists()):
                    # Recorded filename drifted on disk — scan the folder (#291)
                    resolved = reader._scan_storage_for_attachment(
                        attachment_key, attachment["content_type"]
                    )
                if not (resolved and resolved.exists() and resolved.stat().st_size > 0):
                    return None

                # Copy rather than hand back the library path: callers treat
                # the returned file as a scratch copy and delete it, which on
                # a linked file would destroy the user's original (#372).
                shutil.copyfile(resolved, target_path)
                return AttachmentDownloadResult(
                    path=target_path,
                    source="Local storage",
                    errors=errors,
                )
        except Exception as exc:
            errors.append(f"Local storage: {exc}")
            _cleanup_target()

        return None

    def _try_dump(label: str, zot_client: zotero.Zotero | None) -> AttachmentDownloadResult | None:
        if zot_client is None:
            return None

        try:
            zot_client.dump(attachment_key, filename=target_name, path=str(destination))
            if target_path.exists() and target_path.stat().st_size > 0:
                return AttachmentDownloadResult(
                    path=target_path,
                    source=label,
                    errors=errors,
                )
            errors.append(f"{label}: file was not created")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
        finally:
            _cleanup_target()

        return None

    storage_result = _try_local_storage()
    if storage_result:
        return storage_result

    local_result = _try_dump("Local Zotero", local_client)
    if local_result:
        return local_result

    if enable_webdav:
        try:
            webdav_path = download_attachment_from_webdav(
                attachment_key,
                destination,
                expected_filename=target_name,
            )
            if webdav_path.exists() and webdav_path.stat().st_size > 0:
                return AttachmentDownloadResult(
                    path=webdav_path,
                    source="WebDAV",
                    errors=errors,
                )
            errors.append("WebDAV: downloaded file was empty")
        except WebDAVNotConfiguredError:
            pass
        except Exception as exc:
            errors.append(f"WebDAV: {exc}")

    web_result = _try_dump("Web API", web_client)
    if web_result:
        return web_result

    return AttachmentDownloadResult(path=None, source=None, errors=errors)


def convert_to_markdown(file_path: str | Path, *, max_pages: int | None = None) -> str:
    """
    Convert a downloaded attachment to markdown.

    Args:
        file_path: Path to the file to convert.
        max_pages: For PDFs, extract only the first N pages.

    Returns:
        Markdown text, or a human-readable error string on failure.
    """
    doc = extract_file(file_path, max_pages=max_pages)
    if doc is None:
        return f"Error converting file to markdown: {Path(file_path).name}"
    return doc.text
