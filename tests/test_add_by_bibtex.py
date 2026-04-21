"""Integration tests for the zotero_add_by_bibtex MCP tool."""

from conftest import FakeZotero

from zotero_mcp import server


class FakeZoteroWithAttach(FakeZotero):
    """FakeZotero + attachment_both stub (for OA PDF attempts)."""

    def __init__(self):
        super().__init__()
        self.attachments = []

    def attachment_both(self, files, parentid=None, **kwargs):
        self.attachments.append({"files": files, "parentid": parentid})
        return {"success": {"0": "ATCH0001"}, "successful": {}, "failed": {}}


def _patch_hybrid(monkeypatch):
    """Install a write-capable FakeZotero; return it."""
    fake = FakeZoteroWithAttach()
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client",
        lambda ctx: (fake, fake),
    )
    return fake


def _disable_oa_pdf(monkeypatch):
    """Disable network lookups in _try_attach_oa_pdf for tests that create DOIs."""
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._try_attach_oa_pdf",
        lambda *args, **kwargs: "no open-access PDF found (stubbed)",
    )


# ---------------------------------------------------------------------------
# Happy path: single entry
# ---------------------------------------------------------------------------

class TestSingleEntry:
    def test_creates_journal_article(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        bib = """
        @article{smith2020,
          title={Hello},
          author={Smith, John},
          journal={Nature},
          year={2020},
          doi={10.1234/x},
        }
        """
        result = server.add_by_bibtex(bibtex=bib, ctx=dummy_ctx)

        assert len(fake.created) == 1
        created = fake.created[0]
        assert created["itemType"] == "journalArticle"
        assert created["title"] == "Hello"
        assert created["publicationTitle"] == "Nature"
        assert created["DOI"] == "10.1234/x"
        assert "Citation Key: smith2020" in created["extra"]
        assert "Successfully added" in result
        assert "KEY0000" in result

    def test_citekey_preserved_in_extra(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)
        bib = "@book{MyCite2021, title={B}, author={A, B}, publisher={P}, year=2021}"

        server.add_by_bibtex(bibtex=bib, ctx=dummy_ctx)

        assert "Citation Key: MyCite2021" in fake.created[0]["extra"]


# ---------------------------------------------------------------------------
# Multiple entries
# ---------------------------------------------------------------------------

class TestMultipleEntries:
    def test_creates_multiple_items(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        bib = """
        @article{a, title={A}, author={X, Y}, year={2020}, doi={10.1234/a}}
        @book{b, title={B}, author={P, Q}, publisher={Pub}, year={2021}}
        """
        result = server.add_by_bibtex(bibtex=bib, ctx=dummy_ctx)

        assert len(fake.created) == 2
        assert fake.created[0]["itemType"] == "journalArticle"
        assert fake.created[1]["itemType"] == "book"
        assert "Added 2/2 items" in result


# ---------------------------------------------------------------------------
# Caller tags and collections are merged with source tags
# ---------------------------------------------------------------------------

class TestTagsAndCollections:
    def test_caller_tags_merged_with_source_keywords(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        bib = """
        @article{x, title={T}, author={A, B}, year={2020},
          keywords={source1, source2}}
        """
        server.add_by_bibtex(bibtex=bib, tags=["caller1", "source1"], ctx=dummy_ctx)

        tags = [t["tag"] for t in fake.created[0]["tags"]]
        # source1 should only appear once (case-insensitive dedup)
        assert tags == ["source1", "source2", "caller1"]

    def test_collections_applied(self, monkeypatch, dummy_ctx):
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        bib = "@article{x, title={T}, author={A, B}, year={2020}}"
        server.add_by_bibtex(
            bibtex=bib,
            collections=["COL001", "COL002"],
            ctx=dummy_ctx,
        )

        assert fake.created[0]["collections"] == ["COL001", "COL002"]


# ---------------------------------------------------------------------------
# DOI triggers OA PDF attempt; no DOI skips it
# ---------------------------------------------------------------------------

class TestOaPdfAttempt:
    def test_doi_triggers_oa_attempt(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        called = {"count": 0, "doi": None}

        def stub(write_zot, item_key, doi, ctx, **kwargs):
            called["count"] += 1
            called["doi"] = doi
            return "stubbed attempt"

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._try_attach_oa_pdf", stub
        )

        bib = "@article{a, title={T}, author={A, B}, year=2020, doi={10.1234/x}}"
        server.add_by_bibtex(bibtex=bib, ctx=dummy_ctx)

        assert called["count"] == 1
        assert called["doi"] == "10.1234/x"

    def test_no_doi_no_oa_attempt(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        called = {"count": 0}

        def stub(*args, **kwargs):
            called["count"] += 1
            return "stubbed"

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._try_attach_oa_pdf", stub
        )

        bib = "@book{b, title={T}, author={A, B}, publisher={P}, year=2020}"
        server.add_by_bibtex(bibtex=bib, ctx=dummy_ctx)

        assert called["count"] == 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_empty_bibtex(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        result = server.add_by_bibtex(bibtex="", ctx=dummy_ctx)
        assert "No BibTeX" in result

    def test_no_valid_entries(self, monkeypatch, dummy_ctx):
        _patch_hybrid(monkeypatch)
        result = server.add_by_bibtex(bibtex="this is not bibtex", ctx=dummy_ctx)
        assert "No valid @entries" in result

    def test_local_only_mode_rejected(self, monkeypatch, dummy_ctx):
        def raise_local(ctx):
            raise ValueError("Cannot perform write operations in local-only mode.")

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", raise_local
        )
        result = server.add_by_bibtex(bibtex="@a{x, title=T}", ctx=dummy_ctx)
        assert "local-only" in result.lower()

    def test_partial_failure_continues(self, monkeypatch, dummy_ctx):
        """If one entry fails conversion, others should still be created."""
        fake = _patch_hybrid(monkeypatch)
        _disable_oa_pdf(monkeypatch)

        # Monkeypatch create_items to fail on the second call
        call_count = {"n": 0}
        original_create = fake.create_items

        def flaky_create(items, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated write failure")
            return original_create(items, **kwargs)

        fake.create_items = flaky_create

        bib = """
        @article{a, title={A}, author={X, Y}, year={2020}}
        @article{b, title={B}, author={P, Q}, year={2020}}
        @article{c, title={C}, author={R, S}, year={2020}}
        """
        result = server.add_by_bibtex(bibtex=bib, ctx=dummy_ctx)

        assert "Added 2/3 items" in result
        assert "simulated write failure" in result
