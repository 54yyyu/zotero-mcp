"""Live cross-backend parity: zotero_search_items / zotero_advanced_search
must return the same items whether served by the local Zotero-desktop API,
the Zotero web API, or the opt-in SQLite backend (ZOTERO_SEARCH_BACKEND=sqlite).

Gated by ZOTERO_MCP_LIVE_TESTS=1 (see conftest.py) — makes real network/DB
calls against whatever Zotero library is actually connected, using values
discovered at runtime (tests/live/_discovery.py) rather than hardcoded
titles/authors/collections, so the suite runs unmodified against any
tester's library.

Realistic backend-pair coverage per environment:
  - Only local Zotero desktop running: local_api + sqlite are compared;
    web_api tests skip.
  - Only web credentials set: nothing to compare against (a single backend
    has no parity partner), so every test skips.
  - Both available: local_api, web_api, and sqlite are compared pairwise.
All comparisons scope to the connected account's personal library
(group_id=0 / the active library from ZOTERO_LIBRARY_ID/TYPE).

advanced_search's pyzotero fallback has no server-side query support — it
pages the entire library client-side and filters in Python (the exact
slowness the SQL backend exists to fix). Comparing it against a real,
sizeable library can take minutes rather than seconds, so
test_advanced_search_condition_parity skips the api-backend comparison
above _MAX_LIBRARY_SIZE_FOR_API_FALLBACK, keeping this suite practical to
run routinely; test_search_items_free_text_parity is unaffected since
search_items's variant search uses the API's own server-side `q=` filter.
"""

import re

import pytest

from ._discovery import CONDITION_FIELDS

_ITEM_KEY_RE = re.compile(r"\*\*Item Key:\*\*\s*(\w+)")

# advanced_search's pyzotero fallback pages 100 items/request with no
# server-side filter — keep this comparison to libraries small enough that
# the full page-through finishes quickly (see module docstring).
_MAX_LIBRARY_SIZE_FOR_API_FALLBACK = 300


def _backend_pairs(available_backends: dict) -> list[str]:
    return [name for name, client in available_backends.items() if client is not None]


def _run_advanced_search(backend: str, field: str, value: str, monkeypatch,
                          dummy_ctx, *, local_zot, web_zot) -> set[str]:
    from zotero_mcp.tools.search import advanced_search

    if backend == "sqlite":
        monkeypatch.setattr("zotero_mcp.utils.get_search_backend", lambda: "sqlite")
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: local_zot)
    else:
        client = local_zot if backend == "local_api" else web_zot
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: client)

    text = advanced_search(
        conditions=[{"field": field, "operation": "is", "value": value}],
        join_mode="all",
        limit=200,
        ctx=dummy_ctx,
    )
    return set(_ITEM_KEY_RE.findall(text))


def _run_search_items(backend: str, query: str, *, local_zot, web_zot, sql_reader) -> set[str]:
    from zotero_mcp.client import get_active_group_id
    from zotero_mcp.tools.search import _search_with_variants

    if backend == "sqlite":
        result = sql_reader.search_items_sql(
            query, qmode="titleCreatorYear", item_type="-attachment",
            tag=None, limit=200, group_id=get_active_group_id(),
        )
        result = result or []
    else:
        zot = local_zot if backend == "local_api" else web_zot
        result = _search_with_variants(zot, query, "titleCreatorYear", 200)
    return {item.get("key", "") for item in result if item.get("key")}


@pytest.mark.timeout(90)
@pytest.mark.parametrize("field", CONDITION_FIELDS)
def test_advanced_search_condition_parity(field, available_backends, discovered_values,
                                           personal_library_item_count, monkeypatch, dummy_ctx):
    value = discovered_values.get(field)
    if value is None:
        pytest.skip(f"connected library has no usable '{field}' data to test with")

    names = _backend_pairs(available_backends)
    if (personal_library_item_count is not None
            and personal_library_item_count > _MAX_LIBRARY_SIZE_FOR_API_FALLBACK):
        # advanced_search's pyzotero fallback pages the whole library
        # client-side (see module docstring) — too slow to compare routinely
        # against a real-sized library, so only non-API-fallback backends
        # (i.e. sqlite) are compared here.
        names = [n for n in names if n not in ("local_api", "web_api")]
    if len(names) < 2:
        pytest.skip(
            "need >=2 live backends configured to compare parity "
            f"(personal library has {personal_library_item_count} items; "
            f"api-backend comparison skipped above {_MAX_LIBRARY_SIZE_FOR_API_FALLBACK})"
        )

    local_zot = available_backends.get("local_api")
    web_zot = available_backends.get("web_api")

    results = {
        name: _run_advanced_search(name, field, value, monkeypatch, dummy_ctx,
                                    local_zot=local_zot, web_zot=web_zot)
        for name in names
    }
    first_name, first_keys = next(iter(results.items()))
    for name, keys in list(results.items())[1:]:
        assert keys == first_keys, (
            f"advanced_search condition field={field!r} value={value!r} diverged: "
            f"{first_name}={sorted(first_keys)} vs {name}={sorted(keys)}"
        )


@pytest.mark.timeout(60)
def test_search_items_free_text_parity(available_backends, discovered_values):
    query = discovered_values.get("query")
    if query is None:
        pytest.skip("connected library has no item with a usable title to search for")

    names = _backend_pairs(available_backends)
    if len(names) < 2:
        pytest.skip("need >=2 live backends configured to compare parity")

    local_zot = available_backends.get("local_api")
    web_zot = available_backends.get("web_api")
    sql_reader = available_backends.get("sqlite")

    results = {
        name: _run_search_items(name, query, local_zot=local_zot,
                                 web_zot=web_zot, sql_reader=sql_reader)
        for name in names
    }
    first_name, first_keys = next(iter(results.items()))
    for name, keys in list(results.items())[1:]:
        assert keys == first_keys, (
            f"search_items query={query!r} diverged: "
            f"{first_name}={sorted(first_keys)} vs {name}={sorted(keys)}"
        )
