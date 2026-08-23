"""Gating and backend-probe fixtures for the live cross-backend parity suite.

Everything under tests/live/ is skipped unless ZOTERO_MCP_LIVE_TESTS=1 is set
— zero network calls happen otherwise, including at collection time. When
enabled, these fixtures probe whichever real Zotero backends are actually
reachable from this machine (local desktop API, web API, local zotero.sqlite)
and hand back None for anything unavailable, so tests can skip gracefully
rather than fail when e.g. only one backend is configured.
"""

import os

import pytest

LIVE_TESTS_ENV_VAR = "ZOTERO_MCP_LIVE_TESTS"


def pytest_collection_modifyitems(config, items):
    if os.environ.get(LIVE_TESTS_ENV_VAR, "").strip() == "1":
        return
    skip = pytest.mark.skip(
        reason=f"set {LIVE_TESTS_ENV_VAR}=1 to run live cross-backend parity tests"
    )
    for item in items:
        if "tests/live" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def local_zot():
    """Real pyzotero client against local Zotero desktop, or None if
    unreachable (desktop not running, local server disabled, etc.)."""
    from zotero_mcp.client import get_local_zotero_client

    return get_local_zotero_client()


@pytest.fixture(scope="session")
def web_zot():
    """Real pyzotero client against the Zotero web API, or None if
    ZOTERO_LIBRARY_ID/ZOTERO_API_KEY aren't set."""
    from zotero_mcp.client import get_web_zotero_client

    return get_web_zotero_client()


@pytest.fixture(scope="session")
def sql_reader(local_zot):
    """LocalZoteroReader against this machine's zotero.sqlite, or None.

    Only meaningful when local Zotero is set up here (same machine, same
    data directory as `local_zot` just probed) — get_local_zotero_reader()
    additionally requires ZOTERO_LOCAL to be truthy even though it only
    touches the DB file, never the HTTP API, so it's set for the session
    if the caller hasn't already set it.
    """
    if local_zot is None:
        return None
    os.environ.setdefault("ZOTERO_LOCAL", "true")
    from zotero_mcp.local_db import get_local_zotero_reader

    reader = get_local_zotero_reader()
    yield reader
    if reader is not None:
        reader.close()


@pytest.fixture(scope="session")
def available_backends(local_zot, web_zot, sql_reader):
    """{'local_api': zot|None, 'web_api': zot|None, 'sqlite': reader|None}."""
    backends = {"local_api": local_zot, "web_api": web_zot, "sqlite": sql_reader}
    present = [name for name, client in backends.items() if client is not None]
    missing = [name for name, client in backends.items() if client is None]
    print(f"\n[live parity] available backends: {present or 'none'}"
          f"{f' (skipped: {missing})' if missing else ''}")
    return backends


@pytest.fixture(scope="session")
def personal_library_item_count(local_zot, web_zot) -> int | None:
    """Fast item count (zot.num_items(), a single lightweight request) for
    whichever pyzotero client is available, or None if neither is.

    advanced_search's pyzotero fallback has no server-side query support at
    all — it pages the ENTIRE library client-side, 100 items at a time, and
    filters in Python (this is exactly the slowness the SQL backend exists
    to fix). Against a real library of any size that makes it impractical
    for a routine live-test run, so tests use this count to skip that
    specific comparison rather than hang for minutes.
    """
    zot = local_zot or web_zot
    if zot is None:
        return None
    try:
        return zot.num_items()
    except Exception:
        return None


@pytest.fixture(scope="session")
def discovered_values(local_zot, web_zot):
    """Real, non-hardcoded query values pulled from whichever backend is
    reachable — used to exercise each condition field against real data
    from the connected library instead of a fixed name/title/collection."""
    from ._discovery import discover

    zot = local_zot or web_zot
    if zot is None:
        return {}
    return discover(zot)
