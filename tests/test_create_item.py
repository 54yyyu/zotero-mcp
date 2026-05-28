"""Tests for zotero_create_item (create-from-scratch, arbitrary item types)."""

from unittest.mock import patch

import pytest
from conftest import FakeZotero

from zotero_mcp import server


@pytest.fixture
def patch_write_client():
    """Patch _get_write_client to return one fake for both read and write."""
    zot = FakeZotero()
    with patch(
        "zotero_mcp.tools._helpers._get_write_client", return_value=(zot, zot)
    ):
        yield zot


# ---------------------------------------------------------------------------
# conferencePaper (the LIPIcs / DataCite-DOI use case)
# ---------------------------------------------------------------------------

class TestCreateConferencePaper:

    def test_conference_paper_with_datacite_doi(self, dummy_ctx, patch_write_client):
        """A LIPIcs paper (DataCite DOI, not on CrossRef) can be created
        directly as a conferencePaper with venue/volume/pages."""
        zot = patch_write_client
        result = server.create_item(
            item_type="conferencePaper",
            title="An Interesting AFT Paper",
            doi="10.4230/LIPIcs.AFT.2025.29",
            book_title="7th Conference on Advances in Financial Technologies (AFT 2025)",
            volume="354",
            pages="29:1--29:22",
            ctx=dummy_ctx,
        )

        assert len(zot.created) == 1
        d = zot.created[0]
        assert d["itemType"] == "conferencePaper"
        assert d["DOI"] == "10.4230/LIPIcs.AFT.2025.29"
        # book_title lands in the conferencePaper venue field
        assert d["proceedingsTitle"] == \
            "7th Conference on Advances in Financial Technologies (AFT 2025)"
        assert d["volume"] == "354"
        assert d["pages"] == "29:1--29:22"
        assert "Successfully" in result
        assert "conferencePaper" in result

    def test_conference_extra_fields(self, dummy_ctx, patch_write_client):
        zot = patch_write_client
        server.create_item(
            item_type="conferencePaper",
            title="Paper",
            conference_name="AFT 2025",
            place="Pisa, Italy",
            series="LIPIcs",
            ctx=dummy_ctx,
        )
        d = zot.created[0]
        assert d["conferenceName"] == "AFT 2025"
        assert d["place"] == "Pisa, Italy"
        assert d["series"] == "LIPIcs"

    def test_creators_applied(self, dummy_ctx, patch_write_client):
        zot = patch_write_client
        creators = [
            {"creatorType": "author", "firstName": "Ada", "lastName": "Lovelace"},
            {"creatorType": "author", "firstName": "Alan", "lastName": "Turing"},
        ]
        server.create_item(
            item_type="conferencePaper",
            title="Joint Work",
            creators=creators,
            ctx=dummy_ctx,
        )
        assert zot.created[0]["creators"] == creators

    def test_creators_as_json_string(self, dummy_ctx, patch_write_client):
        zot = patch_write_client
        server.create_item(
            item_type="conferencePaper",
            title="Joint Work",
            creators='[{"creatorType": "author", "name": "Anon"}]',
            ctx=dummy_ctx,
        )
        assert zot.created[0]["creators"][0]["name"] == "Anon"

    def test_tags_and_collections(self, dummy_ctx, patch_write_client):
        zot = patch_write_client
        server.create_item(
            item_type="conferencePaper",
            title="Tagged",
            tags=["defi", "consensus"],
            collections=["COLLKEY1"],
            ctx=dummy_ctx,
        )
        d = zot.created[0]
        assert {t["tag"] for t in d["tags"]} == {"defi", "consensus"}
        assert "COLLKEY1" in d["collections"]

    def test_collection_names_resolved(self, dummy_ctx, patch_write_client):
        zot = patch_write_client
        zot._collections = [{"key": "COLL01", "data": {"name": "My Papers"}}]
        server.create_item(
            item_type="conferencePaper",
            title="Resolved",
            collection_names=["My Papers"],
            ctx=dummy_ctx,
        )
        assert "COLL01" in zot.created[0]["collections"]


# ---------------------------------------------------------------------------
# journalArticle
# ---------------------------------------------------------------------------

class TestCreateJournalArticle:

    def test_journal_article_publication_title(self, dummy_ctx, patch_write_client):
        zot = patch_write_client
        server.create_item(
            item_type="journalArticle",
            title="A Study",
            publication_title="Nature",
            volume="600",
            issue="7",
            pages="1-10",
            ctx=dummy_ctx,
        )
        d = zot.created[0]
        assert d["itemType"] == "journalArticle"
        assert d["publicationTitle"] == "Nature"
        assert d["volume"] == "600"
        assert d["issue"] == "7"
        assert d["pages"] == "1-10"


# ---------------------------------------------------------------------------
# Validation and error handling
# ---------------------------------------------------------------------------

class TestCreateItemValidation:

    def test_title_required(self, dummy_ctx, patch_write_client):
        zot = patch_write_client
        result = server.create_item(
            item_type="conferencePaper",
            title="",
            ctx=dummy_ctx,
        )
        assert zot.created == []
        assert "title" in result.lower()
        assert "required" in result.lower()

    def test_invalid_fields_skipped_not_sent(self, dummy_ctx, patch_write_client):
        """Fields invalid for the type are reported as skipped and never put in
        the payload (Zotero rejects unknown fields)."""
        zot = patch_write_client
        # book_title -> bookTitle is invalid for a journalArticle
        result = server.create_item(
            item_type="journalArticle",
            title="X",
            book_title="Should Not Apply",
            ctx=dummy_ctx,
        )
        d = zot.created[0]
        assert "bookTitle" not in d
        assert "book_title" in result  # skip warning mentions the param

    def test_local_only_rejected(self, dummy_ctx):
        with patch(
            "zotero_mcp.tools._helpers._get_write_client",
            side_effect=ValueError(
                "Cannot perform write operations in local-only mode. "
                "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
            ),
        ):
            result = server.create_item(
                item_type="conferencePaper",
                title="X",
                ctx=dummy_ctx,
            )
        assert "local-only" in result.lower() or "cannot" in result.lower()
