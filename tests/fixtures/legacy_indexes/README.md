# Legacy indexes, written by the real old releases

`pre017.zip` and `v0171.zip` are what zotero-mcp's own `~/.config/zotero-mcp` looked like after
`zotero-mcp update-db --fulltext` on a four-item library, run by the old release itself (not
reconstructed by hand). Each holds `config.json` and `chroma_db/`. `tests/test_legacy_index_fixtures.py`
(and the migration tests built on it) unzip one into a temporary home and run the current code on it.

| Zip | Release (PyPI `zotero-mcp-server`) | Layout |
|---|---|---|
| `pre017.zip` | 0.16.0, the latest 0.16.x | one collection `zotero_library`, chunking on: only passages `KEY#n` with copied metadata, no article records (an item without a file is one passage `KEY#0`) |
| `v0171.zip` | 0.17.1 | `zotero_library`: one article record per item, bare `KEY`; `zotero_library_passages`: passages `KEY#n` with copied metadata |

Both were built with Python 3.11 and chromadb 1.5.9, the latest 1.x when the releases shipped (it
is also the version in `uv.lock`). Chunking: `chunk_size` 400, `overlap` 50, so every file gives
several passages.

## The fake Zotero

`fake_zotero.py` (standard library only) writes `zotero.sqlite` and `storage/<KEY>/<file>` and is
imported by both the generator and the tests, so the library the old release indexed is rebuilt
identically at test time. User library only; all dates fixed.

| Item | Files | Passages (0.16 / 0.17.1) |
|---|---|---|
| `ITEMONE1` Neural networks for soil moisture retrieval | one PDF, `ATTONEA1` | 4 |
| `ITEMTWO2` Bayesian methods in community ecology | two PDFs, `ATTTWOA2` (small) and `ATTTWOB2` (large) | 6, from the large one |
| `ITEMTHR3` A note on lichen taxonomy | none; metadata and abstract only | 1 |
| `ITEMFOU4` Glacier retreat in the Alps since 1850 | one PDF, `ATTFOUR4`, long text | 19 |

The PDFs are minimal hand-written one-font PDFs the extractor reads as text.

## No model, no network

`fake_embedding.py` replaces the embedding model. It registers as `default` (ChromaDB's default
embedding name, which is what the old config says), and returns a 384-dimension hashed bag-of-words
vector, so a query like "glacier retreat" finds the glacier item and nothing is downloaded. It does
not subclass ChromaDB's own `DefaultEmbeddingFunction`, because ChromaDB sets such an instance
aside and embeds with the persisted configuration, which would be the real model. `_build.py` also
blocks sockets and gives the old release an empty Zotero API, so neither the internet nor a running
Zotero is reached.

## Regenerating

    uv run --no-project python tests/fixtures/legacy_indexes/make_legacy_indexes.py

Needs `uv` and PyPI for the installs (nothing else). Each release runs in its own `uv run
--isolated` environment with a clean environment and `HOME` inside a temporary directory; the only
files written to the repository are the two zips. ChromaDB gives its segments random UUIDs, so a
regenerated zip differs byte for byte but not in content. The generator is not run in CI.
