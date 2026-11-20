"""Retrying fulltext extraction: legacy item markers and per-file markers.

Records written before #826 are "legacy" (they carry ``title``). A legacy
item keeps its records until it changes; the criteria are the ones it was
written under: a "failed" has_fulltext marker clears when the item's
attachment set changes (attaching a file does NOT bump the parent's
dateModified) or the item is edited. A changed legacy item is rewritten in
the item-keyed shape: a bibliographic unit plus one unit per selected file.

New records track failures per file instead: a file whose extraction
yielded nothing is not parsed again until its file_stamp changes.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from _scan_fakes import ScanReader, target

from zotero_mcp import fulltext_cache, semantic_search

DATE_MODIFIED = "2026-07-02 01:01:48"


class FakeItem:
    def __init__(self):
        self.item_id = 1
        self.key = "ITEMKEY1"
        self.item_type = "journalArticle"
        self.title = "Calibration"
        self.abstract = ""
        self.extra = ""
        self.doi = None
        self.notes = None
        self.creators = None
        self.date_added = "2026-07-01 00:00:00"
        self.date_modified = DATE_MODIFIED
        self.fulltext = None
        self.fulltext_source = None


class FakeReader(ScanReader):
    """Minimal LocalZoteroReader stand-in for the extraction scan."""

    def __init__(self, *args, attachments=(), texts=None, **kwargs):
        super().__init__(targets={1: [target(key) for key, _p, _c in attachments]}, texts=texts)

    def get_all_item_keys(self):
        return {"ITEMKEY1"}

    def get_key_group_map(self):
        # Fixed result rather than a real query — this file tests
        # failed-marker retry, not library attribution.
        return ({"ITEMKEY1": 0}, set())

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        return [FakeItem()]


class FakeChromaClient:
    def __init__(self, stored_metadata, legacy=True):
        self.embedding_max_tokens = 8000
        # Stored docs here model the post-migration steady state (attributed
        # to the personal library) unless a test says otherwise: a doc with
        # NO group_id is deliberately re-upserted by the scan regardless of
        # the failed marker, which would defeat these skip/retry assertions.
        if stored_metadata is not None:
            stored_metadata = dict(stored_metadata)
            stored_metadata.setdefault("group_id", 0)
            if legacy:
                # Every record written before #826 carried the title.
                stored_metadata.setdefault("title", "Calibration")
        self._stored = stored_metadata

    def get_document_metadata(self, key):
        return self._stored if key == "ITEMKEY1" else None

    def get_existing_ids(self, ids):
        return set()

    def truncate_text(self, text, max_tokens=None):
        return text

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def reset_collection(self):
        pass


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    return str(tmp_path / "config.json")


def _search(monkeypatch, reader, chroma, config_path):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=config_path)
    # A fake-induced error would silently fall back to the API path — fail loudly instead
    monkeypatch.setattr(
        search,
        "_get_items_from_api",
        lambda *a, **kw: pytest.fail("unexpected fallback to API path"),
    )
    return search


def _run_scan(monkeypatch, stored_metadata, attachments, config_path, texts=None, legacy=True):
    reader = FakeReader(attachments=attachments, texts=texts)
    chroma = FakeChromaClient(stored_metadata, legacy=legacy)
    search = _search(monkeypatch, reader, chroma, config_path)
    items = search._get_items_from_source(extract_fulltext=True, chroma_client=chroma, force_rebuild=False)
    return items, reader, search


def _run_scan_no_text(monkeypatch, stored_metadata, attachments, config_path, legacy=True):
    texts = {key: None for key, _p, _c in attachments}
    return _run_scan(monkeypatch, stored_metadata, attachments, config_path, texts=texts, legacy=legacy)


def _keys(items):
    return [(item["key"], item["unit"]) for item in items]


PDF = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]


def test_failed_item_skipped_when_nothing_changed(monkeypatch, config_path):
    """failed marker + same date + same attachment set -> no retry."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
    }
    items, reader, _ = _run_scan(monkeypatch, stored, [], config_path)
    assert items == []
    assert reader.extract_calls == 0


def test_untagged_doc_is_reupserted_despite_failed_marker(monkeypatch, config_path):
    """A doc with no group_id predates multi-library attribution and was not
    covered by the backfill; the scan must re-upsert it so it gains its
    attribution — otherwise it is skipped as 'up to date' forever and stays
    excluded from library-filtered search and deletion cleanup."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
    }
    reader = FakeReader(attachments=[])
    chroma = FakeChromaClient(stored)
    chroma._stored.pop("group_id", None)
    search = _search(monkeypatch, reader, chroma, config_path)
    search.db_path = "unused"

    items = search._get_items_from_local_db(extract_fulltext=True, chroma_client=chroma)

    assert [it["key"] for it in items] == ["ITEMKEY1"]


def test_failed_item_retried_when_attachment_added(monkeypatch, config_path):
    """failed marker + same date, but a PDF was attached since -> retry.

    The changed legacy item is rewritten by unit: its metadata, and the new
    file under its own key with only its own state (#826).
    """
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
    }
    items, reader, search = _run_scan(monkeypatch, stored, PDF, config_path)
    assert _keys(items) == [("ITEMKEY1", "item"), ("ATTKEY1", "attachment")]
    assert items[0]["legacy_rewrite"] is True
    assert reader.extract_calls == 1
    assert items[1]["text"] == "extracted text"
    records = search._build_item_records(items[1], chunking=False)
    assert records[0]["metadata"]["file_stamp"] == "1:100"
    assert "has_fulltext" not in records[0]["metadata"]


def test_failed_item_retried_when_date_modified_changed(monkeypatch, config_path):
    """failed marker + same attachment set, but the item was edited -> retry.

    Pins the pre-existing dateModified trigger ("user replaces a bad PDF and
    the parent gets re-saved") so it survives the new attachment-set check.
    """
    stored = {
        "has_fulltext": "failed",
        "date_modified": "2026-06-30 12:00:00",  # differs from the item's
        "attachment_keys": "ATTKEY1",
    }
    items, reader, _ = _run_scan(monkeypatch, stored, PDF, config_path)
    assert _keys(items) == [("ITEMKEY1", "item"), ("ATTKEY1", "attachment")]
    assert reader.extract_calls == 1


def test_legacy_failed_record_without_attachment_keys_retries_once(monkeypatch, config_path):
    """Records written before attachment_keys existed retry once, then converge."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        # no attachment_keys field (legacy)
    }
    items, reader, _ = _run_scan(monkeypatch, stored, PDF, config_path)
    assert len(items) == 2
    assert reader.extract_calls == 1


# ---------------------------------------------------------------------------
# #446: "failed" must mean an extraction actually failed
# ---------------------------------------------------------------------------


def test_item_without_attachments_is_not_marked_failed(monkeypatch, config_path):
    """No attachments -> nothing was tried -> no failure recorded (#446)."""
    items, reader, search = _run_scan_no_text(monkeypatch, None, [], config_path)
    assert _keys(items) == [("ITEMKEY1", "item")]
    assert reader.extract_calls == 0
    assert "has_fulltext" not in search._create_metadata(items[0])


def test_item_with_unreadable_attachment_is_marked_failed(monkeypatch, config_path):
    """A file that produced no text is a real failure: marked at its stamp."""
    items, reader, _ = _run_scan_no_text(monkeypatch, None, PDF, config_path)
    assert _keys(items) == [("ITEMKEY1", "item")]
    assert fulltext_cache.failed("ATTKEY1", "1:100", config_path=config_path)


def test_failed_file_is_not_parsed_again_until_it_changes(monkeypatch, config_path):
    _run_scan_no_text(monkeypatch, None, PDF, config_path)

    items, reader, _ = _run_scan(monkeypatch, None, PDF, config_path)

    assert _keys(items) == [("ITEMKEY1", "item")]
    assert reader.extract_calls == 0


def test_attachmentless_item_never_reported_as_failed_extraction(monkeypatch, config_path, capsys):
    """Legacy 'failed' markers on attachment-less items leave the report.

    Records written before #446 carry has_fulltext="failed" with an empty
    attachment_keys set; they are still skipped (nothing to do), but the
    summary must not present them as extraction failures.
    """
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
    }
    items, reader, _search = _run_scan_no_text(monkeypatch, stored, [], config_path)
    assert items == []
    assert reader.extract_calls == 0
    err = capsys.readouterr().err
    assert "extraction previously failed" not in err


def test_real_failed_extraction_still_reported(monkeypatch, config_path, capsys):
    """Files whose extraction failed at their current stamp show in the report."""
    fulltext_cache.record_failure("ATTKEY1", "1:100", config_path=config_path)
    stored = {"date_modified": DATE_MODIFIED, "item_key": "ITEMKEY1"}
    items, reader, _search = _run_scan_no_text(monkeypatch, stored, PDF, config_path, legacy=False)
    assert items == []
    assert reader.extract_calls == 0
    err = capsys.readouterr().err
    assert "extraction previously failed" in err


def test_extraction_summary_counts_failures_in_the_same_run(monkeypatch, config_path, capsys):
    """The run that fails an extraction says so itself, not just the next one."""
    _run_scan_no_text(monkeypatch, None, PDF, config_path)
    err = capsys.readouterr().err
    assert "1 attachment(s) produced no text" in err


# ---------------------------------------------------------------------------
# #428: indexed fulltext must follow the item's attachments
# ---------------------------------------------------------------------------


def test_item_whose_attachments_were_removed_is_reindexed_metadata_only(monkeypatch, config_path):
    """Text indexed from a PDF the user has since deleted must stop matching."""
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "ATTKEY1",
    }
    items, reader, search = _run_scan_no_text(monkeypatch, stored, [], config_path)
    assert _keys(items) == [("ITEMKEY1", "item")]
    # Its legacy passages, cut from the deleted file, go once it is written.
    assert items[0]["legacy_rewrite"] is True
    assert "has_fulltext" not in search._create_metadata(items[0])


def test_replaced_attachment_is_reextracted(monkeypatch, config_path):
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "OLDATT01",
    }
    attachments = [("NEWATT01", "storage:new.pdf", "application/pdf")]
    items, reader, _ = _run_scan(monkeypatch, stored, attachments, config_path)
    assert _keys(items) == [("ITEMKEY1", "item"), ("NEWATT01", "attachment")]
    assert reader.extract_calls == 1


def test_unchanged_indexed_item_is_still_skipped(monkeypatch, config_path):
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "ATTKEY1",
    }
    items, reader, _ = _run_scan(monkeypatch, stored, PDF, config_path)
    assert items == []
    assert reader.extract_calls == 0


def test_legacy_record_without_attachment_keys_is_not_reextracted(monkeypatch, config_path):
    """Upgrading must not re-extract every document indexed before
    attachment_keys was stored."""
    stored = {"has_fulltext": True, "date_modified": DATE_MODIFIED}
    items, reader, _ = _run_scan(monkeypatch, stored, PDF, config_path)
    assert items == []
    assert reader.extract_calls == 0
