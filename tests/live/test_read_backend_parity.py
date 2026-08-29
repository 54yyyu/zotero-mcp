"""Live parity: SqliteBackend and ApiBackend must answer identically.

`test_search_backend_parity.py` pins this down for the two search tools.
This file extends it to the rest of the read port — items, children,
collections, tags, recent — because those operations moved off pyzotero
onto `library.LibraryBackend`, and the only thing standing between "one
query instead of 174" and "one query that quietly returns something else"
is a comparison against the backend that was there before.

Needs BOTH backends to be live, so it skips unless Zotero desktop is
running (local API) or web credentials are set, *and* zotero.sqlite is
readable here. That is the opposite requirement from
`test_sqlite_only_reads.py`, which needs the APIs to be *un*reachable —
the two files deliberately cover opposite environments, and on any given
machine one of them will usually skip.

Comparisons are on item keys and names rather than whole records: the two
backends legitimately differ on fields the API computes server-side (the
`meta` block, `numChildren`), and asserting on those would fail for the
wrong reason. Where field-level fidelity matters it is asserted
explicitly, on the fields the tools actually render.
"""

import os

import pytest


@pytest.fixture(scope="session")
def both_backends(local_zot, web_zot):
    """(sqlite_backend, api_backend) or a skip when either is unavailable."""
    zot = local_zot or web_zot
    if zot is None:
        pytest.skip("no Zotero API reachable — parity needs a backend to compare against")

    os.environ.setdefault("ZOTERO_LOCAL", "true")
    from zotero_mcp.client import get_active_group_id
    from zotero_mcp.library import ApiBackend, SqliteBackend
    from zotero_mcp.local_db import get_local_zotero_reader

    reader = get_local_zotero_reader()
    if reader is None:
        pytest.skip("no readable zotero.sqlite on this machine")
    try:
        yield SqliteBackend(reader, get_active_group_id()), ApiBackend(zot)
    finally:
        reader.close()


@pytest.fixture(scope="session")
def sample_keys(both_backends):
    """A handful of real top-level item keys both backends can see."""
    sqlite_backend, _ = both_backends
    items = sqlite_backend.recent_items(limit=8)
    keys = [item["key"] for item in items if item.get("key")]
    if not keys:
        pytest.skip("library has no items to compare")
    return keys


def _keys(items) -> set[str]:
    return {i.get("key") for i in items or [] if i.get("key")}


def test_get_items_agree_on_membership(both_backends, sample_keys):
    sqlite_backend, api_backend = both_backends
    from_sqlite = sqlite_backend.get_items(sample_keys)
    from_api = api_backend.get_items(sample_keys)
    assert set(from_sqlite) == set(from_api), (
        f"item lookup diverged: sqlite={sorted(from_sqlite)} api={sorted(from_api)}"
    )


@pytest.mark.parametrize(
    "field", ["itemType", "title", "date", "DOI", "publicationTitle", "abstractNote"]
)
def test_get_items_agree_on_rendered_fields(both_backends, sample_keys, field):
    """The fields the read tools actually render must match exactly.

    Parametrized per field so a divergence names which one broke rather
    than dumping two whole records.
    """
    sqlite_backend, api_backend = both_backends
    from_sqlite = sqlite_backend.get_items(sample_keys)
    from_api = api_backend.get_items(sample_keys)

    for key in sorted(set(from_sqlite) & set(from_api)):
        sqlite_value = from_sqlite[key]["data"].get(field, "")
        api_value = from_api[key]["data"].get(field, "")
        assert sqlite_value == api_value, (
            f"{key}.{field} diverged: sqlite={sqlite_value!r} api={api_value!r}"
        )


def test_get_items_agree_on_creators(both_backends, sample_keys):
    sqlite_backend, api_backend = both_backends
    from_sqlite = sqlite_backend.get_items(sample_keys)
    from_api = api_backend.get_items(sample_keys)

    def names(item):
        return [
            (c.get("creatorType"), c.get("lastName") or c.get("name"))
            for c in item["data"].get("creators", [])
        ]

    for key in sorted(set(from_sqlite) & set(from_api)):
        assert names(from_sqlite[key]) == names(from_api[key]), f"{key} creators diverged"


def test_get_children_agree(both_backends, sample_keys):
    """The 3281x optimisation must return the same children it replaced."""
    sqlite_backend, api_backend = both_backends
    from_sqlite = sqlite_backend.get_children(sample_keys)
    from_api = api_backend.get_children(sample_keys)

    for key in sample_keys:
        assert _keys(from_sqlite.get(key)) == _keys(from_api.get(key)), (
            f"children of {key} diverged: "
            f"sqlite={sorted(_keys(from_sqlite.get(key)))} "
            f"api={sorted(_keys(from_api.get(key)))}"
        )


def test_list_collections_agree(both_backends):
    sqlite_backend, api_backend = both_backends
    from_sqlite = {c["key"]: c["data"].get("name") for c in sqlite_backend.list_collections()}
    from_api = {c["key"]: c["data"].get("name") for c in api_backend.list_collections()}
    assert set(from_sqlite) == set(from_api), "collection membership diverged"
    for key in from_sqlite:
        assert from_sqlite[key] == from_api[key], f"collection {key} name diverged"


def test_list_tags_agree(both_backends):
    """The 67x optimisation must list the same tags the OFFSET walk did."""
    sqlite_backend, api_backend = both_backends
    assert set(sqlite_backend.list_tags()) == set(api_backend.list_tags())


def test_collection_items_agree(both_backends):
    sqlite_backend, api_backend = both_backends
    collections = sqlite_backend.list_collections()
    if not collections:
        pytest.skip("library has no collections")
    for coll in collections:
        key = coll["key"]
        from_sqlite = sqlite_backend.collection_items(key)
        if from_sqlite:
            break
    else:
        pytest.skip("library has no non-empty collection")
    from_api = api_backend.collection_items(key)
    assert _keys(from_sqlite) == _keys(from_api), (
        f"collection {key} items diverged"
    )


def test_get_item_reports_trash_the_same_way(both_backends, sample_keys):
    """`data["deleted"]` drives the trash status the tools render."""
    sqlite_backend, api_backend = both_backends
    from_sqlite = sqlite_backend.get_items(sample_keys)
    from_api = api_backend.get_items(sample_keys)
    for key in sorted(set(from_sqlite) & set(from_api)):
        assert bool(from_sqlite[key]["data"].get("deleted")) == bool(
            from_api[key]["data"].get("deleted")
        ), f"{key} trash status diverged"
