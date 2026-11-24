"""A search costs one Chroma query and one Zotero round in the common case.

A paper's own record and its file's record collapse into one result, so a
first fetch of exactly ``limit`` records comes back short and asks again. The
first fetch is sized for that collapse. Web-mode hits on an attachment's
record guess their parent from ``parent_hint`` in the same batch.

Fakes only; HOME and Path.home point into tmp_path.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

from test_semantic_owner_resolution import (  # noqa: E402
    _attachment,
    _hit,
    _isolated,  # noqa: F401  (autouse fixture)
    _keys,
    _regular,
    _search,
)


def _papers_with_files(n):
    hits, items = [], []
    for i in range(n):
        paper, att = f"PAP{i:05d}", f"ATT{i:05d}"
        hits.append(_hit(paper, "the abstract", item_key=paper))
        hits.append(_hit(f"{att}#0", "the file text", item_key=att, parent_hint=paper, chunk_index=0, n_chunks=1))
        items += [_regular(paper), _attachment(att, paper)]
    return hits, items


@pytest.mark.parametrize("chunking", [False, True])
def test_an_unfiltered_search_with_collapsing_records_is_one_round(monkeypatch, tmp_path, chunking):
    hits, items = _papers_with_files(40)
    s, zot = _search(monkeypatch, tmp_path, hits, items, chunking=chunking)

    out = s.search("mindfulness", limit=10)

    assert _keys(out) == [f"PAP{i:05d}" for i in range(10)]
    assert len(s.chroma_client.calls) == 1
    assert len(zot.batch_calls) == 1


def test_chunked_hits_name_their_parent_in_the_first_batch(monkeypatch, tmp_path):
    hits = [_hit(f"ATT{i}#0", "text", item_key=f"ATT{i}", parent_hint=f"PAP{i}", chunk_index=0, n_chunks=1)
            for i in range(5)]
    items = [x for i in range(5) for x in (_regular(f"PAP{i}"), _attachment(f"ATT{i}", f"PAP{i}"))]
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=5)

    assert _keys(out) == [f"PAP{i}" for i in range(5)]
    assert len(zot.batch_calls) == 1


def test_a_wrong_hint_still_resolves_with_a_second_call(monkeypatch, tmp_path):
    hits = [
        _hit("ATT1#0", "text", item_key="ATT1", parent_hint="PAP1", chunk_index=0, n_chunks=1),
        _hit("ATT2#0", "text", item_key="ATT2", parent_hint="WRONG", chunk_index=0, n_chunks=1),
        _hit("ATT3#0", "text", item_key="ATT3", chunk_index=0, n_chunks=1),
    ]
    items = [_regular("PAP1"), _regular("PAP2"), _regular("PAP3"),
             _attachment("ATT1", "PAP1"), _attachment("ATT2", "PAP2"), _attachment("ATT3", "PAP3")]
    s, zot = _search(monkeypatch, tmp_path, hits, items)

    out = s.search("q", limit=5)

    assert _keys(out) == ["PAP1", "PAP2", "PAP3"]
    assert len(zot.batch_calls) == 2
    assert set(zot.batch_calls[1]) == {"PAP2", "PAP3"}
