"""An item's full text is kept only as far as its records can use it.

A scan holds the text of every item it is about to index, which for a first
build is the whole library: up to 50 pages a PDF, of which an article record
embeds the model's token limit and passages the first ``max_chunks_per_item``
chunks. The part nothing reads is dropped when the text is read; the records
built from it must be the same.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from test_fulltext_web_mode import FakeChromaClient, FakeZoteroClient, _build_search, _paper, _pdf
from test_passage_chunking import _chunking_search, _fit_search

SENTENCE = "Mindfulness training reduced relapse in the treated group. "


def _item(text):
    return {"key": "ITEM0001", "data": {"key": "ITEM0001", "itemType": "journalArticle",
                                        "title": "A long paper", "fulltext": text}}


def test_the_cap_is_the_longer_of_what_an_article_and_the_passages_read(monkeypatch):
    search = _fit_search(monkeypatch, 256, max_chunks_per_item=20)   # the default model
    chunk = search._effective_chunk_size()
    assert search._fulltext_char_cap() == max(256 * 16, chunk * 20) == chunk * 20

    search._chunking_config["enabled"] = False
    assert search._fulltext_char_cap() == 256 * 16

    search.chroma_client.embedding_max_tokens = None
    assert search._fulltext_char_cap() is None


@pytest.mark.parametrize("chunking", [True, False])
def test_records_built_from_capped_text_are_the_same(monkeypatch, chunking):
    search = _fit_search(monkeypatch, 256, max_chunks_per_item=20)
    search._chunking_config["enabled"] = chunking
    # An article record is cut by the model's tokenizer, here four characters a token.
    search.chroma_client.truncate_text = lambda text, max_tokens=None: text[: 256 * 4]
    text = SENTENCE * 4000
    cap = search._fulltext_char_cap()
    assert len(text) > 5 * cap

    whole = search._build_item_records(_item(text), chunking=chunking, article=True)
    capped = search._build_item_records(_item(text[:cap]), chunking=chunking, article=True)

    assert len(whole) == (21 if chunking else 1)
    assert capped == whole


def test_text_from_zotero_is_cut_to_the_cap(monkeypatch):
    zot = FakeZoteroClient()
    attachment = _pdf("ATT1", parent="P1")
    zot.load_scenario(
        [_paper("P1"), attachment],
        fulltext={"ATT1": {"content": SENTENCE * 4000}},
        children={"P1": [attachment]},
    )
    chroma = FakeChromaClient()
    chroma.embedding_max_tokens = 256
    search = _build_search(monkeypatch, zot, chroma)
    paper = _paper("P1")

    search._attach_web_fulltext([paper])

    assert len(paper["data"]["fulltext"]) == 256 * 16
    assert paper["data"]["fulltext"] == (SENTENCE * 4000)[: 256 * 16]


def test_short_text_is_left_alone(monkeypatch):
    search = _chunking_search(monkeypatch)
    assert search._fulltext_char_cap() >= 8000 * 16


def test_text_extracted_by_the_local_scan_is_cut_to_the_cap(monkeypatch):
    from test_failed_marker_retry import FakeChromaClient as ScanChroma
    from test_failed_marker_retry import FakeReader

    from zotero_mcp import semantic_search

    class LongTextReader(FakeReader):
        def extract_fulltext_for_item(self, item_id):
            return (SENTENCE * 4000, "pdf")

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    reader = LongTextReader(attachments=[("ATT1", "storage:paper.pdf", "application/pdf")])
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
    chroma = ScanChroma(None)
    chroma.embedding_max_tokens = 256
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    monkeypatch.setattr(
        search, "_get_items_from_api", lambda *a, **kw: pytest.fail("unexpected fallback to API path")
    )

    items = search._get_items_from_source(extract_fulltext=True, chroma_client=chroma, force_rebuild=False)

    assert items[0]["data"]["fulltext"] == (SENTENCE * 4000)[: 256 * 16]
    assert items[0]["data"]["fulltextSource"] == "pdf"
