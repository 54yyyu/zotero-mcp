"""Tests for the shims that let one call site drive either Zotero backend."""

import json

import pytest
from conftest import DummyContext, FakeLocalZotero, _FakeResponse
from pyzotero.errors import CallDoesNotExistError, LocalAPIKeyRequiredError

from zotero_mcp import client as _client
from zotero_mcp.tools import _helpers


@pytest.fixture(autouse=True)
def _clear_template_cache():
    _helpers._local_template_cache.clear()
    yield
    _helpers._local_template_cache.clear()


class TestItemTemplateFor:
    def test_uses_the_endpoint_when_it_exists(self, fake_zot):
        template = _helpers.item_template_for(fake_zot, "journalArticle")
        assert template["itemType"] == "journalArticle"
        assert "publicationTitle" in template

    def test_passes_the_link_mode_through(self, fake_zot):
        calls = []
        fake_zot.item_template = (
            lambda t, linkmode=None: calls.append((t, linkmode)) or {"itemType": t}
        )
        _helpers.item_template_for(fake_zot, "attachment", "linked_url")
        assert calls == [("attachment", "linked_url")]

    def test_synthesizes_when_items_new_is_missing(self, fake_local_zot):
        template = _helpers.item_template_for(fake_local_zot, "journalArticle")
        assert template["itemType"] == "journalArticle"
        # Built from item_type_fields, which the local API does implement.
        assert template["title"] == ""
        assert template["abstractNote"] == ""
        # Universal keys no field endpoint reports.
        assert template["creators"] == []
        assert template["collections"] == []
        assert template["relations"] == {}

    def test_attachment_templates_match_the_web_response(self, fake_local_zot):
        imported = _helpers.item_template_for(fake_local_zot, "attachment", "imported_file")
        assert imported["linkMode"] == "imported_file"
        assert set(imported) == {
            "itemType", "linkMode", "title", "accessDate", "note", "tags",
            "collections", "relations", "contentType", "charset", "filename",
            "md5", "mtime",
        }

        linked = _helpers.item_template_for(fake_local_zot, "attachment", "linked_url")
        assert linked["linkMode"] == "linked_url"
        assert "url" in linked
        assert "filename" not in linked

    def test_note_template(self, fake_local_zot):
        assert _helpers.item_template_for(fake_local_zot, "note") == {
            "itemType": "note", "note": "", "tags": [], "collections": [], "relations": {},
        }

    def test_synthesized_templates_are_cached(self, fake_local_zot):
        calls = []
        fake_local_zot.item_type_fields = lambda t: calls.append(t) or [{"field": "title"}]
        for _ in range(3):
            _helpers.item_template_for(fake_local_zot, "journalArticle")
        assert calls == ["journalArticle"]

    def test_callers_cannot_corrupt_the_cache(self, fake_local_zot):
        first = _helpers.item_template_for(fake_local_zot, "journalArticle")
        first["title"] = "mutated"
        second = _helpers.item_template_for(fake_local_zot, "journalArticle")
        assert second["title"] == ""

    def test_unrelated_errors_are_not_swallowed(self, fake_zot):
        def _boom(item_type, link_mode=None):
            raise RuntimeError("network down")

        fake_zot.item_template = _boom
        with pytest.raises(RuntimeError, match="network down"):
            _helpers.item_template_for(fake_zot, "journalArticle")


class TestAttachFiles:
    def test_uses_attachment_both_on_the_web_api(self, fake_zot, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        result = _helpers.attach_files(fake_zot, [("Paper", str(pdf))], parentid="P1")
        assert result["success"] == [{"key": "ATT00001"}]
        assert fake_zot.attached == [([("Paper", str(pdf))], "P1")]
        assert fake_zot.uploaded == []

    def test_uploads_hand_built_items_on_the_local_api(self, fake_local_zot, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        result = _helpers.attach_files(fake_local_zot, [("Paper", str(pdf))], parentid="P1")
        assert result["success"] == [{"key": "ATT00001"}]

        (payload, parentid) = fake_local_zot.uploaded[0]
        assert parentid == "P1"
        attachment = payload[0]
        assert attachment["title"] == "Paper"
        assert attachment["linkMode"] == "imported_file"
        assert attachment["contentType"] == "application/pdf"
        # Zupload resolves filename against basedir to read the bytes, so it
        # needs the path, not the display name.
        assert attachment["filename"] == str(pdf)

    def test_unknown_extension_falls_back_to_octet_stream(self, fake_local_zot, tmp_path):
        blob = tmp_path / "data.zzz"
        blob.write_bytes(b"x")
        _helpers.attach_files(fake_local_zot, [("Data", str(blob))])
        assert fake_local_zot.uploaded[0][0][0]["contentType"] == "application/octet-stream"

    def test_both_backends_return_the_same_shape(self, fake_zot, fake_local_zot, tmp_path):
        pdf = tmp_path / "p.pdf"
        pdf.write_bytes(b"%PDF")
        web = _helpers.attach_files(fake_zot, [("P", str(pdf))])
        local = _helpers.attach_files(fake_local_zot, [("P", str(pdf))])
        assert set(web) == set(local) == {"success", "unchanged", "failure"}


class TestWebdavSkip:
    def test_skipped_for_local_uploads(self, monkeypatch, fake_local_zot):
        """Bytes uploaded through the local API are already in Zotero's own
        storage; the desktop syncs them to WebDAV itself."""
        monkeypatch.setattr("zotero_mcp.webdav.is_webdav_configured", lambda: True)
        result = _helpers._maybe_upload_to_webdav(
            {"success": [{"key": "ATT1"}]}, "/tmp/x.pdf", DummyContext(),
            write_zot=fake_local_zot,
        )
        assert result == ""

    def test_still_runs_for_web_uploads(self, monkeypatch, fake_zot):
        monkeypatch.setattr("zotero_mcp.webdav.is_webdav_configured", lambda: True)
        uploaded = []
        monkeypatch.setattr(
            "zotero_mcp.webdav.upload_attachment_to_webdav",
            lambda attachment_key, file_path: uploaded.append(attachment_key),
        )
        result = _helpers._maybe_upload_to_webdav(
            {"success": [{"key": "ATT1"}]}, "/tmp/x.pdf", DummyContext(),
            write_zot=fake_zot,
        )
        assert uploaded == ["ATT1"]
        assert "uploaded to WebDAV" in result


class TestTrashItem:
    def _item(self):
        return {"key": "ITEM01", "version": 7, "data": {"key": "ITEM01"}}

    def test_routes_through_write_so_the_headers_are_attached(self, fake_local_zot):
        ok, detail = _helpers.trash_item(fake_local_zot, self._item())
        assert (ok, detail) == (True, "")
        method, url, kwargs = fake_local_zot.writes[0]
        assert method == "PATCH"
        assert url.endswith("/users/0/items/ITEM01")
        assert kwargs["headers"]["If-Unmodified-Since-Version"] == "7"
        assert json.loads(kwargs["content"]) == {"deleted": 1}

    def test_falls_back_to_client_patch_without_write(self):
        """Keeps working against clients (and test doubles) that predate the
        _write dispatcher."""
        calls = []

        class _Legacy:
            local = False
            endpoint = "https://api.zotero.org"
            library_type = "users"
            library_id = "123"

            class client:
                @staticmethod
                def patch(url, headers, content):
                    calls.append((url, headers, content))
                    return _FakeResponse(204)

        ok, _detail = _helpers.trash_item(_Legacy(), self._item())
        assert ok is True
        assert len(calls) == 1

    def test_maps_a_local_401_to_advice(self):
        """_write returns the response without raising, so the status has to
        be interpreted here."""
        zot = FakeLocalZotero(write_status=401)
        ok, detail = _helpers.trash_item(zot, self._item())
        assert ok is False
        assert "Always Allow" in detail

    def test_reports_a_version_conflict_plainly(self):
        zot = FakeLocalZotero(write_status=412)
        ok, detail = _helpers.trash_item(zot, self._item())
        assert ok is False
        assert "changed in Zotero" in detail


class TestErrorMessages:
    def test_local_key_required(self):
        msg = _helpers.format_zotero_error(LocalAPIKeyRequiredError("401"))
        assert "single-use" in msg
        assert "authorize-local" in msg

    def test_a_generic_401_from_a_local_client_still_gets_advice(self, fake_local_zot):
        """pyzotero picks the specific class by matching the response body, so
        a 401 whose body doesn't match arrives as a plain error."""
        msg = _helpers.describe_write_failure(_FakeResponse(401), fake_local_zot)
        assert "Always Allow" in msg

    def test_a_401_from_the_web_api_is_left_alone(self, fake_zot):
        msg = _helpers.describe_write_failure(_FakeResponse(401, "Forbidden"), fake_zot)
        assert "Always Allow" not in msg
        assert "401" in msg

    def test_server_id_mismatch_drops_the_unusable_key(self, fake_local_zot):
        _client.store_local_write_credentials("stale", "old-srv", remember=True)
        msg = _helpers.describe_write_failure(
            _FakeResponse(412, "Zotero-Server-ID does not match"), fake_local_zot
        )
        assert "different Zotero database" in msg
        assert _client.get_local_write_credentials() == (None, None, None)

    def test_missing_server_id_reads_as_a_bug(self, fake_local_zot):
        msg = _helpers.describe_write_failure(_FakeResponse(428), fake_local_zot)
        assert "Internal error" in msg

    def test_single_use_note_is_added_when_we_know(self, fake_local_zot, monkeypatch):
        monkeypatch.setattr(_client, "_local_write_state",
                            {"key": "k", "server_id": "srv", "remember": False})
        msg = _helpers.describe_write_failure(_FakeResponse(401), fake_local_zot)
        assert "granted with \"Allow\"" in msg

    def test_a_rejected_key_is_dropped_so_the_next_write_can_fall_back(
        self, fake_local_zot
    ):
        """A local 401 means the key is dead. Keeping it would outrank working
        web credentials on every subsequent write."""
        _client.store_local_write_credentials("dead", "srv", remember=True)
        _helpers.describe_write_failure(_FakeResponse(401), fake_local_zot)
        assert _client.get_local_write_credentials() == (None, None, None)

    def test_points_at_the_web_fallback_when_one_exists(self, fake_local_zot, monkeypatch):
        monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
        _client.store_local_write_credentials("dead", "srv", remember=True)
        msg = _helpers.describe_write_failure(_FakeResponse(401), fake_local_zot)
        assert "next write will use those" in msg

    def test_asks_for_reauthorization_when_there_is_no_fallback(
        self, fake_local_zot, monkeypatch
    ):
        monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
        monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
        _client.store_local_write_credentials("dead", "srv", remember=True)
        msg = _helpers.describe_write_failure(_FakeResponse(401), fake_local_zot)
        assert "authorize-local" in msg

    def test_unknown_errors_pass_through_unchanged(self):
        assert _helpers.format_zotero_error(ValueError("something else")) == "something else"

    def test_missing_items_new_endpoint_should_never_surface(self, fake_local_zot):
        msg = _helpers.format_zotero_error(
            CallDoesNotExistError("no /items/new"), fake_local_zot
        )
        assert "report" in msg
