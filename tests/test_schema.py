"""Tests for the base-field schema resolver (zotero_mcp.schema).

These exercise the resolver against the *real* vendored table so they also
guard the shipped data file, plus the refresh/fallback behaviour with a stubbed
HTTP layer (no network).
"""

import json
import time

import pytest

from zotero_mcp import schema


@pytest.fixture(autouse=True)
def _isolate_table_memo():
    """refresh() mutates the module-level table memo; keep tests independent."""
    saved = schema._table_cache
    schema._table_cache = None
    yield
    schema._table_cache = saved


# ---------------------------------------------------------------------------
# Resolution: generic/base param -> the type's actual field key
# ---------------------------------------------------------------------------

class TestResolveField:

    def test_base_title_routes_to_type_specific_key(self):
        assert schema.resolve_field("statute", "title") == "nameOfAct"
        assert schema.resolve_field("case", "title") == "caseName"
        assert schema.resolve_field("email", "title") == "subject"

    def test_base_date_routes_to_type_specific_key(self):
        assert schema.resolve_field("statute", "date") == "dateEnacted"
        assert schema.resolve_field("case", "date") == "dateDecided"

    def test_native_field_is_unchanged(self):
        # journalArticle stores title as literal "title"
        assert schema.resolve_field("journalArticle", "title") == "title"

    def test_publicationTitle_routes_per_type(self):
        assert schema.resolve_field("bookSection", "publicationTitle") == "bookTitle"
        assert schema.resolve_field("journalArticle", "publicationTitle") == "publicationTitle"

    def test_unknown_field_returned_unchanged(self):
        assert schema.resolve_field("statute", "definitelyNotAField") == "definitelyNotAField"

    def test_unknown_item_type_returns_field_unchanged(self):
        assert schema.resolve_field("notARealType", "title") == "title"


# ---------------------------------------------------------------------------
# Validation: is a (resolved) field valid for this type?
# ---------------------------------------------------------------------------

class TestValidFields:

    def test_includes_type_specific_key(self):
        assert "nameOfAct" in schema.valid_fields("statute")

    def test_excludes_base_name_when_renamed(self):
        # statute has nameOfAct, not a literal "title"
        assert "title" not in schema.valid_fields("statute")

    def test_includes_universal_citationKey(self):
        # retires the #321 citationKey special-case: it's a real schema field
        assert "citationKey" in schema.valid_fields("statute")
        assert "citationKey" in schema.valid_fields("journalArticle")

    def test_unknown_type_has_empty_field_set(self):
        assert schema.valid_fields("notARealType") == set()


# ---------------------------------------------------------------------------
# Inverse (for the deferred read-side fix)
# ---------------------------------------------------------------------------

class TestBaseFieldOf:

    def test_inverts_rename(self):
        assert schema.base_field_of("statute", "nameOfAct") == "title"
        assert schema.base_field_of("case", "caseName") == "title"

    def test_native_field_is_its_own_base(self):
        assert schema.base_field_of("journalArticle", "title") == "title"


# ---------------------------------------------------------------------------
# Refresh: TTL-gated conditional GET, outcome signalling, never fatal
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


def _seed_cache(path, *, version=42, etag='"v42"', checked=0):
    path.write_text(json.dumps({
        "version": version,
        "itemTypes": {"statute": {"nameOfAct": "title"}},
        "_etag": etag,
        "_checked": checked,
    }), encoding="utf-8")


class TestRefresh:

    def test_offline_returns_offline_and_keeps_resolution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))

        def _boom(url, headers):
            raise OSError("no network")

        monkeypatch.setattr(schema, "_http_get", _boom)
        assert schema.refresh(force=True) == "offline"
        # resolution still works from the vendored floor
        assert schema.resolve_field("statute", "title") == "nameOfAct"

    def test_malformed_200_returns_offline_without_raising(self, tmp_path, monkeypatch):
        """A 200 with a non-JSON body (captive portal / proxy) must not raise."""
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))

        class _BadBody:
            status_code = 200
            headers = {}

            def json(self):
                raise ValueError("Expecting value: line 1 column 1")

        monkeypatch.setattr(schema, "_http_get", lambda url, headers: _BadBody())
        assert schema.refresh(force=True) == "offline"
        assert schema.resolve_field("statute", "title") == "nameOfAct"

    def test_500_returns_offline_and_leaves_cache_intact(self, tmp_path, monkeypatch):
        cache = tmp_path / "schema.json"
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(cache))
        _seed_cache(cache, version=42, checked=0)
        monkeypatch.setattr(schema, "_http_get",
                            lambda url, headers: _FakeHTTPResponse(500))
        assert schema.refresh(force=True) == "offline"
        assert json.loads(cache.read_text())["version"] == 42

    def test_200_returns_refreshed_and_updates_memo(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))
        doc = {
            "version": 999,
            "itemTypes": [
                {"itemType": "brandNewType",
                 "fields": [{"field": "coolName", "baseField": "title"}]},
            ],
        }
        monkeypatch.setattr(
            schema, "_http_get",
            lambda url, headers: _FakeHTTPResponse(200, doc, {"ETag": '"v999"'}),
        )
        assert schema.refresh(force=True) == "refreshed"
        # memo updated — no table passed, resolution reflects the new schema
        assert schema.get_table()["version"] == 999
        assert schema.resolve_field("brandNewType", "title") == "coolName"

    def test_200_without_etag_stores_none(self, tmp_path, monkeypatch):
        cache = tmp_path / "schema.json"
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(cache))
        doc = {"version": 50, "itemTypes": [
            {"itemType": "statute",
             "fields": [{"field": "nameOfAct", "baseField": "title"}]}]}
        monkeypatch.setattr(schema, "_http_get",
                            lambda url, headers: _FakeHTTPResponse(200, doc, {}))
        assert schema.refresh(force=True) == "refreshed"
        assert json.loads(cache.read_text())["_etag"] is None

    def test_304_returns_unchanged(self, tmp_path, monkeypatch):
        cache = tmp_path / "schema.json"
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(cache))
        _seed_cache(cache, version=42, etag='"v42"', checked=0)

        def _not_modified(url, headers):
            assert headers.get("If-None-Match") == '"v42"'
            return _FakeHTTPResponse(304)

        monkeypatch.setattr(schema, "_http_get", _not_modified)
        assert schema.refresh(force=True) == "unchanged"
        assert schema.resolve_field("statute", "title") == "nameOfAct"

    def test_within_ttl_skips_the_network(self, tmp_path, monkeypatch):
        """The production startup path: a fresh cache must not hit the network."""
        cache = tmp_path / "schema.json"
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(cache))
        _seed_cache(cache, checked=time.time())  # fresh

        def _must_not_call(url, headers):
            raise AssertionError("network called within the TTL")

        monkeypatch.setattr(schema, "_http_get", _must_not_call)
        assert schema.refresh(force=False) == "unchanged"

    def test_stale_ttl_triggers_conditional_get(self, tmp_path, monkeypatch):
        cache = tmp_path / "schema.json"
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(cache))
        _seed_cache(cache, etag='"v42"', checked=0)  # stale
        seen = []

        def _spy(url, headers):
            seen.append(headers.get("If-None-Match"))
            return _FakeHTTPResponse(304)

        monkeypatch.setattr(schema, "_http_get", _spy)
        assert schema.refresh(force=False) == "unchanged"
        assert seen == ['"v42"']


class TestOfflineBackoff:
    """An offline machine must not retry (and warn) on every single startup."""

    def test_failed_refresh_backs_off_on_the_next_startup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))
        calls = []

        def _boom(url, headers):
            calls.append(url)
            raise OSError("no network")

        monkeypatch.setattr(schema, "_http_get", _boom)
        assert schema.refresh(force=False) == "offline"
        assert len(calls) == 1
        # Second startup within the back-off window: no second request.
        assert schema.refresh(force=False) == "offline"
        assert len(calls) == 1
        # An explicit `zotero-mcp schema-refresh` still bypasses the back-off.
        assert schema.refresh(force=True) == "offline"
        assert len(calls) == 2

    def test_backoff_lapses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))
        calls = []

        def _boom(url, headers):
            calls.append(url)
            raise OSError("no network")

        monkeypatch.setattr(schema, "_http_get", _boom)
        assert schema.refresh(force=False) == "offline"
        schema._attempt_path().write_text(
            str(time.time() - schema.FAILED_REFRESH_BACKOFF_SECONDS - 1), encoding="utf-8"
        )
        assert schema.refresh(force=False) == "offline"
        assert len(calls) == 2

    def test_success_clears_the_backoff_marker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))
        monkeypatch.setattr(schema, "_http_get",
                            lambda url, headers: _FakeHTTPResponse(500))
        assert schema.refresh(force=False) == "offline"
        assert schema._attempt_path().exists()

        doc = {"version": 43, "itemTypes": [
            {"itemType": "statute", "fields": [{"field": "nameOfAct", "baseField": "title"}]},
        ]}
        monkeypatch.setattr(schema, "_http_get",
                            lambda url, headers: _FakeHTTPResponse(200, doc, {"ETag": '"v43"'}))
        assert schema.refresh(force=True) == "refreshed"
        assert not schema._attempt_path().exists()


class TestRefreshOptOut:

    def test_env_var_disables_the_network_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_REFRESH", "0")

        def _must_not_call(url, headers):
            raise AssertionError("network called with refresh disabled")

        monkeypatch.setattr(schema, "_http_get", _must_not_call)
        assert schema.refresh(force=False) == "disabled"
        # Resolution is unaffected — the vendored floor is always correct.
        assert schema.resolve_field("statute", "title") == "nameOfAct"

    def test_explicit_refresh_overrides_the_opt_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(tmp_path / "schema.json"))
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_REFRESH", "0")
        doc = {"version": 43, "itemTypes": [
            {"itemType": "statute", "fields": [{"field": "nameOfAct", "baseField": "title"}]},
        ]}
        monkeypatch.setattr(schema, "_http_get",
                            lambda url, headers: _FakeHTTPResponse(200, doc, {"ETag": '"v43"'}))
        assert schema.refresh(force=True) == "refreshed"


class TestCorruptCache:

    def test_truncated_cache_falls_back_to_the_vendored_floor(self, tmp_path, monkeypatch):
        cache = tmp_path / "schema.json"
        monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE", str(cache))
        # Parses as JSON, but a crash/full disk left it without itemTypes.
        cache.write_text('{"version": 42}', encoding="utf-8")
        schema._table_cache = None
        assert schema.resolve_field("statute", "title") == "nameOfAct"
