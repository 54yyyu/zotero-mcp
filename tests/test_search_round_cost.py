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
