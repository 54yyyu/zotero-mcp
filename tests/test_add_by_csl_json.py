"""Integration tests for the zotero_add_by_csl_json MCP tool."""

import json

from conftest import FakeZotero

from zotero_mcp import server


class FakeZoteroWithAttach(FakeZotero):
    def __init__(self):
        super().__init__()
        self.attachments = []

    def attachment_both(self, files, parentid=None, **kwargs):
        self.attachments.append({"files": files, "parentid": parentid})
        return {"success": {"0": "ATCH0001"}, "successful": {}, "failed": {}}


def _patch_hybrid(monkeypatch):
    fake = FakeZoteroWithAttach()
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client",
        lambda ctx: (fake, fake),
    )
    return fake


def _disable_oa_pdf(monkeypatch):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._try_attach_oa_pdf",
        lambda *args, **kwargs: "no open-access PDF found (stubbed)",
    )


SAMPLE_ARTICLE = {
    "type": "article-journal",
    "id": "X2020",
    "title": "A Paper",
    "author": [{"given": "John", "family": "Smith"}],
    "issued": {"date-parts": [[2020, 3, 15]]},
    "container-title": "Nature",
    "volume": "42",
    "issue": "7",
    "page": "1-10",
    "DOI": "10.1234/x",
}


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_single_dict_input(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        result = server.add_by_csl_json(csl_json=SAMPLE_ARTICLE, ctx=dummy_ctx)

        assert len(fake.created) == 1
        created = fake.created[0]
        assert created["itemType"] == "journalArticle"
        assert created["title"] == "A Paper"
        assert created["DOI"] == "10.1234/x"
        assert created["date"] == "2020-03-15"
        assert "Citation Key: X2020" in created["extra"]
        assert "Successfully added" in result

    def test_json_string_input(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        result = server.add_by_csl_json(
            csl_json=json.dumps(SAMPLE_ARTICLE), ctx=dummy_ctx,
        )

        assert len(fake.created) == 1
        assert fake.created[0]["title"] == "A Paper"
        assert "Successfully added" in result

    def test_list_of_objects(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        entries = [SAMPLE_ARTICLE, {"type": "book", "title": "B",
                                     "author": [{"family": "A"}]}]
        result = server.add_by_csl_json(csl_json=entries, ctx=dummy_ctx)

        assert len(fake.created) == 2
        assert fake.created[0]["itemType"] == "journalArticle"
        assert fake.created[1]["itemType"] == "book"
        assert "Added 2/2" in result

    def test_json_string_array(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        server.add_by_csl_json(
            csl_json=json.dumps([SAMPLE_ARTICLE, SAMPLE_ARTICLE]),
            ctx=dummy_ctx,
        )

        assert len(fake.created) == 2


# ---------------------------------------------------------------------------
# Tags and collections
# ---------------------------------------------------------------------------

class TestTagsAndCollections:
    def test_caller_tags_merged(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        csl = dict(SAMPLE_ARTICLE)
        csl["keyword"] = ["source1", "source2"]

        server.add_by_csl_json(
            csl_json=csl, tags=["caller1", "source1"], ctx=dummy_ctx,
        )

        tags = [t["tag"] for t in fake.created[0]["tags"]]
        assert tags == ["source1", "source2", "caller1"]

    def test_collections_applied(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        server.add_by_csl_json(
            csl_json=SAMPLE_ARTICLE,
            collections=["COL001"],
            ctx=dummy_ctx,
        )

        assert fake.created[0]["collections"] == ["COL001"]


# ---------------------------------------------------------------------------
# DOI -> OA PDF
# ---------------------------------------------------------------------------

class TestOaPdfAttempt:
    def test_doi_triggers_attempt(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        called = {"count": 0, "doi": None}

        def stub(write_zot, item_key, doi, ctx, **kwargs):
            called["count"] += 1
            called["doi"] = doi
            return "stubbed"

        monkeypatch.setattr("zotero_mcp.tools._helpers._try_attach_oa_pdf", stub)

        server.add_by_csl_json(csl_json=SAMPLE_ARTICLE, ctx=dummy_ctx)

        assert called["count"] == 1
        assert called["doi"] == "10.1234/x"

    def test_no_doi_no_attempt(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        called = {"count": 0}

        def stub(*args, **kwargs):
            called["count"] += 1
            return "stubbed"

        monkeypatch.setattr("zotero_mcp.tools._helpers._try_attach_oa_pdf", stub)

        server.add_by_csl_json(
            csl_json={"type": "book", "title": "B",
                      "author": [{"family": "A"}]},
            ctx=dummy_ctx,
        )

        assert called["count"] == 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_invalid_json_string(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        result = server.add_by_csl_json(csl_json="{not valid json", ctx=dummy_ctx)
        assert "Invalid JSON" in result

    def test_empty_list(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        result = server.add_by_csl_json(csl_json=[], ctx=dummy_ctx)
        assert "No valid CSL JSON" in result

    def test_empty_string(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        result = server.add_by_csl_json(csl_json="", ctx=dummy_ctx)
        assert "No valid CSL JSON" in result

    def test_local_only_mode_rejected(self, monkeypatch, dummy_ctx):
        def raise_local(ctx):
            raise ValueError("Cannot perform write operations in local-only mode.")

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", raise_local
        )
        result = server.add_by_csl_json(csl_json=SAMPLE_ARTICLE, ctx=dummy_ctx)
        assert "local-only" in result.lower()
