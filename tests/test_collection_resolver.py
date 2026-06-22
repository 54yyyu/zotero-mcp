"""Tests for resolve_collection_specs / build_collection_paths (#3).

The resolver lets every add path (and zotero_manage_collections) accept
collection KEYS, NAMES, or '/'-separated PATHS interchangeably:

- a key is used as-is, but only if it actually exists as a live collection
  (shape alone is not enough — trashed/bogus keys must fail loudly);
- a name matches case-insensitively anywhere in the tree;
- a path like 'parent/child' suffix-matches the collection tree, so it
  disambiguates same-named leaves;
- ambiguity and misses raise ValueError with actionable messages;
- create_missing=True creates the missing chain instead of raising.
"""

import pytest

from conftest import DummyContext, FakeZotero
from zotero_mcp.tools import _helpers


def _coll(key, name, parent=False):
    return {"key": key, "data": {"name": name, "parentCollection": parent}}


class FakeZoteroResolver(FakeZotero):
    """FakeZotero with collection-creation tracking and a stubbed trash."""

    def __init__(self):
        super().__init__()
        self.created_collections = []
        self._trashed = []
        self._create_counter = 0

    def create_collections(self, colls, **kwargs):
        self.created_collections.extend(colls)
        result = {}
        for i, c in enumerate(colls):
            self._create_counter += 1
            new_key = f"NEW{self._create_counter:05d}"
            result[str(i)] = new_key
            self._collections.append(
                _coll(new_key, c["name"], c.get("parentCollection") or False)
            )
        return {"success": result, "successful": {}, "failed": {}}

    def _retrieve_data(self, path):
        class _Resp:
            def __init__(self, data):
                self._data = data

            def json(self):
                return self._data

        if path.endswith("/collections/trash"):
            return _Resp(self._trashed)
        raise Exception(f"unexpected path {path}")


@pytest.fixture
def zot():
    z = FakeZoteroResolver()
    z._collections = [
        _coll("ROOT0001", "Machine Learning"),
        _coll("CHLD0001", "Deep Learning", parent="ROOT0001"),
        _coll("CHLD0002", "Optimisation", parent="ROOT0001"),
        _coll("ROOT0002", "_project"),
        _coll("CHLD0003", "Deep Learning", parent="ROOT0002"),  # duplicate leaf name
        _coll("ROOT0003", "Reading List"),
    ]
    return z


# ---------------------------------------------------------------------------
# build_collection_paths
# ---------------------------------------------------------------------------

class TestBuildCollectionPaths:
    def test_paths_built_from_parent_links(self, zot):
        paths = _helpers.build_collection_paths(zot._collections)
        assert paths["ROOT0001"] == ["Machine Learning"]
        assert paths["CHLD0001"] == ["Machine Learning", "Deep Learning"]
        assert paths["CHLD0003"] == ["_project", "Deep Learning"]

    def test_orphaned_parent_degrades_to_short_path(self):
        paths = _helpers.build_collection_paths(
            [_coll("AAAA0001", "Orphan", parent="GONE0000")]
        )
        assert paths["AAAA0001"] == ["Orphan"]

    def test_parent_cycle_does_not_hang(self):
        paths = _helpers.build_collection_paths([
            _coll("AAAA0001", "A", parent="BBBB0001"),
            _coll("BBBB0001", "B", parent="AAAA0001"),
        ])
        assert "A" in "/".join(paths["AAAA0001"])
        assert "B" in "/".join(paths["BBBB0001"])


# ---------------------------------------------------------------------------
# resolve_collection_specs — keys
# ---------------------------------------------------------------------------

class TestKeyResolution:
    def test_live_key_passes_through(self, zot):
        assert _helpers.resolve_collection_specs(zot, ["CHLD0001"]) == ["CHLD0001"]

    def test_bogus_key_shaped_spec_errors(self, zot):
        with pytest.raises(ValueError, match="not found"):
            _helpers.resolve_collection_specs(zot, ["DEADBEEF"])

    def test_trashed_key_reports_trash(self, zot):
        zot._trashed = [_coll("DEAD0001", "Old Stuff")]
        with pytest.raises(ValueError, match="Trash"):
            _helpers.resolve_collection_specs(zot, ["DEAD0001"])

    def test_key_shaped_name_resolves_as_name(self, zot):
        # 8-char uppercase name that is NOT a live key → name resolution.
        zot._collections.append(_coll("ROOT0004", "ARCHIVES"))
        assert _helpers.resolve_collection_specs(zot, ["ARCHIVES"]) == ["ROOT0004"]


# ---------------------------------------------------------------------------
# resolve_collection_specs — names and paths
# ---------------------------------------------------------------------------

class TestNameAndPathResolution:
    def test_exact_name_case_insensitive(self, zot):
        assert _helpers.resolve_collection_specs(zot, ["machine learning"]) == ["ROOT0001"]

    def test_nested_name_matches_anywhere(self, zot):
        assert _helpers.resolve_collection_specs(zot, ["Optimisation"]) == ["CHLD0002"]

    def test_ambiguous_name_lists_candidates(self, zot):
        with pytest.raises(ValueError) as exc:
            _helpers.resolve_collection_specs(zot, ["Deep Learning"])
        msg = str(exc.value)
        assert "ambiguous" in msg
        assert "CHLD0001" in msg and "CHLD0003" in msg
        assert "Machine Learning/Deep Learning" in msg

    def test_path_disambiguates(self, zot):
        assert _helpers.resolve_collection_specs(
            zot, ["_project/Deep Learning"]
        ) == ["CHLD0003"]

    def test_path_is_case_insensitive(self, zot):
        assert _helpers.resolve_collection_specs(
            zot, ["machine learning/deep learning"]
        ) == ["CHLD0001"]

    def test_unknown_name_suggests_close_matches(self, zot):
        with pytest.raises(ValueError) as exc:
            _helpers.resolve_collection_specs(zot, ["learning"])
        # 'learning' is a substring of several collections but an exact match
        # of none → ambiguous-or-missing must surface candidates.
        msg = str(exc.value)
        assert "not found" in msg
        assert "Machine Learning" in msg

    def test_mixed_specs_resolve_in_order_and_dedupe(self, zot):
        out = _helpers.resolve_collection_specs(
            zot, ["ROOT0003", "reading list", "Optimisation"]
        )
        assert out == ["ROOT0003", "CHLD0002"]

    def test_empty_specs_no_fetch(self):
        class Boom:
            def collections(self, **kw):
                raise AssertionError("should not fetch for empty specs")

        assert _helpers.resolve_collection_specs(Boom(), []) == []
        assert _helpers.resolve_collection_specs(Boom(), None) == []


# ---------------------------------------------------------------------------
# resolve_collection_specs — create_missing
# ---------------------------------------------------------------------------

class TestCreateMissing:
    def test_creates_single_name_at_root(self, zot):
        out = _helpers.resolve_collection_specs(
            zot, ["Brand New"], create_missing=True, write_zot=zot,
            ctx=DummyContext(),
        )
        assert out == ["NEW00001"]
        assert zot.created_collections == [
            {"name": "Brand New", "parentCollection": False}
        ]

    def test_creates_chain_under_existing_prefix(self, zot):
        out = _helpers.resolve_collection_specs(
            zot, ["_project/intertemporal/drafts"], create_missing=True,
            write_zot=zot, ctx=DummyContext(),
        )
        # '_project' exists (ROOT0002); 'intertemporal' and 'drafts' created.
        assert [c["name"] for c in zot.created_collections] == [
            "intertemporal", "drafts",
        ]
        assert zot.created_collections[0]["parentCollection"] == "ROOT0002"
        assert out == ["NEW00002"]

    def test_existing_spec_not_recreated(self, zot):
        out = _helpers.resolve_collection_specs(
            zot, ["Reading List"], create_missing=True, write_zot=zot,
        )
        assert out == ["ROOT0003"]
        assert zot.created_collections == []

    def test_ambiguous_parent_prefix_errors(self, zot):
        with pytest.raises(ValueError, match="ambiguous"):
            _helpers.resolve_collection_specs(
                zot, ["Deep Learning/subtopic"], create_missing=True,
                write_zot=zot,
            )

    def test_create_missing_without_write_client_errors(self, zot):
        with pytest.raises(ValueError, match="writable"):
            _helpers.resolve_collection_specs(
                zot, ["Brand New"], create_missing=True,
            )
