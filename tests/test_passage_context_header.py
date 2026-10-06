"""Study-context headers on passages, and keeping them out of what is shown.

With ``chunking.context_header`` on, passages after the first are embedded as
``"<header>\\n\\n<passage>"`` so a sentence from the middle of a paper is
matched as part of that paper. ``_passage_body`` recovers the passage for
display without storing it twice.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search
from zotero_mcp.semantic_search import _passage_body

HEADER = "Learning from experience (2023)\nBion, Wilfred; Hinshelwood, Robert\nAn abstract."


def test_header_is_stripped_from_later_passages():
    body = "full designation of the function is alpha-function."
    doc = f"{HEADER}\n\n{body}"
    meta = {"chunk_index": 3, "char_start": 1000, "char_end": 1000 + len(body), "title": "Learning from experience"}
    assert _passage_body(doc, meta) == body


def test_stripping_tolerates_a_passage_trimmed_by_strip():
    # The splitter strips each window, so the stored span can exceed the text.
    body = "a passage"
    doc = f"{HEADER}\n\n{body}"
    meta = {"chunk_index": 1, "char_start": 0, "char_end": len(body) + 3, "title": "Learning from experience"}
    assert _passage_body(doc, meta) == body


def test_abstract_with_blank_line_does_not_cut_early():
    header = "Title (2020)\nAuthor\nFirst abstract paragraph.\n\nSecond abstract paragraph."
    body = "The passage proper."
    doc = f"{header}\n\n{body}"
    meta = {"chunk_index": 2, "char_start": 10, "char_end": 10 + len(body), "title": "Title"}
    assert _passage_body(doc, meta) == body


def test_first_passage_is_untouched():
    doc = "Title\nAuthor\n\nAbstract text and the opening of the paper."
    meta = {"chunk_index": 0, "char_start": 0, "char_end": len(doc), "title": "Title"}
    assert _passage_body(doc, meta) == doc


def test_passage_with_no_header_is_untouched():
    # An annotation or untitled item is indexed without a header.
    doc = "plain passage text\n\nwith a paragraph break"
    meta = {"chunk_index": 2, "char_start": 0, "char_end": len(doc), "title": ""}
    assert _passage_body(doc, meta) == doc


def test_header_must_start_with_the_title():
    doc = "Something else\n\nshort"
    meta = {"chunk_index": 2, "char_start": 0, "char_end": 5, "title": "Real Title"}
    assert _passage_body(doc, meta) == doc


def test_missing_offsets_leave_text_alone():
    doc = f"{HEADER}\n\nbody"
    assert _passage_body(doc, {"chunk_index": 2}) == doc
    assert _passage_body(doc, None) == doc
    assert _passage_body("", {"chunk_index": 1}) == ""


def test_enrichment_shows_the_passage_not_the_header(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=object())
    s._attach_zotero_items = lambda enriched: None
    body = "Alpha-function transforms sense impressions into elements usable for thinking."
    doc = f"{HEADER}\n\n{body}"
    meta = {"chunk_index": 5, "char_start": 40, "char_end": 40 + len(body), "title": "Learning from experience"}
    out = s._enrich_search_results(
        {"ids": [["BION#5"]], "distances": [[0.2]], "documents": [[doc]], "metadatas": [[meta]]},
        "Learning from experience alpha-function",
        limit=5,
    )
    assert out[0]["matched_text"] == body
    assert "Hinshelwood" not in out[0]["matched_passage"]


# --- indexing ---------------------------------------------------------------


class _RecordingChroma:
    embedding_max_tokens = 8000

    def __init__(self):
        self.docs, self.metas, self.ids = [], [], []

    def get_existing_ids(self, ids):
        return set()

    def delete_item_chunks(self, item_key):
        pass

    def prune_item_chunks(self, item_key, keep):
        pass

    def upsert_documents(self, documents, metadatas, ids):
        self.docs.extend(documents)
        self.metas.extend(metadatas)
        self.ids.extend(ids)

    def truncate_text(self, text, max_tokens=None):
        return text


def _index(monkeypatch, item, **chunking):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    chroma = _RecordingChroma()
    s = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    s._chunking_config = {
        "enabled": True, "chunk_size": 200, "overlap": 20, "max_chunks_per_item": 10, **chunking,
    }
    s._process_item_batch([item], force_rebuild=True)
    return chroma


def _paper(item_type="journalArticle", title="Learning from experience"):
    return {
        "key": "BION0001",
        "data": {
            "title": title,
            "itemType": item_type,
            "date": "2023",
            "abstractNote": "Bion's theory of thinking.",
            "creators": [{"creatorType": "author", "firstName": "Wilfred", "lastName": "Bion"}],
            "fulltext": "Alpha-function turns sense data into thought. " * 40,
        },
    }


def test_header_is_off_by_default(monkeypatch):
    chroma = _index(monkeypatch, _paper())
    assert len(chroma.docs) > 2
    assert not any(doc.startswith("Learning from experience (2023)") for doc in chroma.docs[1:])


def test_header_prefixes_later_passages_when_enabled(monkeypatch):
    chroma = _index(monkeypatch, _paper(), context_header=True)
    assert len(chroma.docs) > 2
    for doc, meta in zip(chroma.docs[1:], chroma.metas[1:]):
        assert doc.startswith("Learning from experience (2023)\nBion, Wilfred")
        # Offsets still address the original document, and the body is recoverable.
        assert _passage_body(doc, meta) == doc.split("\n\n", 1)[1]
    # Passage 0 opens with the structured text already; no second header.
    assert not chroma.docs[0].startswith("Learning from experience (2023)\nBion")


def test_untitled_items_get_no_header(monkeypatch):
    chroma = _index(monkeypatch, _paper(title=""), context_header=True)
    assert all("\n\nAlpha-function" not in doc[:60] for doc in chroma.docs[1:])
