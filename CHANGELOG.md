# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-03-22

### Changed
- Updated README to remove outdated fork section and reflect current feature set.

## [0.1.5] - 2026-03-22

### Added
- **Write operations** — 10+ new tools: `create_item`, `update_item`, `create_note`, `add_tags`, `batch_update_tags`, `create_collection`, `add_to_collection`, `remove_from_collection`, `add_by_doi`, `add_by_url`, `add_from_file` (PR #165).
- **BetterBibTeX citation key lookup** — `search_by_citation_key` searches both BetterBibTeX JSON-RPC and the Extra field (#72).
- **PDF outline extraction** — `get_pdf_outline` returns table of contents from PDFs.
- **Annotation page labels** — `get_annotations` now includes `annotationPageLabel` and `annotationPosition` data (#159).
- **PDF timeout** — configurable `pdf_timeout` (default 30s) skips slow PDFs during fulltext extraction (#74).
- **Semantic search quality** — combined field+fulltext embeddings, Gemini `retrieval_query`/`retrieval_document` fix, model-aware tokenizer, optional cross-encoder re-ranking (PR #154).
- **Abstracts in collection items** — `get_collection_items` now includes abstracts (#143).
- **Local-first fulltext extraction** — prefers local DB/storage before remote `dump()` for file-backed attachments (PR #166).
- **`--fulltext` guard** — aborts with clear error when used without `ZOTERO_LOCAL` enabled (PR #156).

### Fixed
- **search_notes** — fixed `qmode` and client-side filter to actually find notes (#137).
- **batch_update_tags** — fixed stale tag set, response type check, and added hybrid local+web mode (#162).
- **get_tags pagination** — uses `zot.everything()` for reliable tag retrieval (#70).
- **Fulltext truncation** — removed hardcoded 10k/5k char caps; model-aware truncation via `embedding_max_tokens` (#153, #134).
- **Local mode file:// paths** — resolves `file://`, absolute paths, and `attachments:` prefixes (#116).
- **Child notes** — `create_note` properly attaches as child via web API in local mode (#133).
- **ChromaDB embedding conflict** — auto-detects and resets collection on model change (#109).
- **FastMCP compatibility** — removed deprecated `dependencies` parameter (#117, #61).
- **PDF outline import order** — defers PyMuPDF import until after attachment check.
- **Update interval display** — fixed misleading display for daily schedule (PR #144).
- **Config loading** — embedding model config now loads correctly from config file (#76).

## [0.1.4] - 2026-03-09

### Added
- Model-aware token truncation for embedding models.

### Fixed
- Truncate documents to embedding model token limit to prevent failures with large texts.
- Search notes now correctly finds notes by content.
- Note creation properly attaches notes as child items via web API.
- Auto-reset ChromaDB collection on embedding model change.
- Updated default Gemini model to `gemini-embedding-001`.
- Implemented `get_config`/`build_from_config` for ChromaDB embedding functions.
- Fixed test `FakeChromaClient` missing `embedding_max_tokens` attribute.

## [0.1.3] - 2026-02-20

### Changed
- Published to PyPI as `zotero-mcp-server`. Install with `pip install zotero-mcp-server`.
- Updater now checks PyPI for latest versions (with GitHub releases as fallback).
- Updater now installs/upgrades from PyPI instead of git URLs.
- Install instructions updated to use PyPI in README and docs.

### Added
- PyPI badge in README.
- `keywords`, `license`, and additional `project.urls` metadata in package config.
- This changelog.

### Fixed
- Cleaned up `MANIFEST.in` (removed reference to nonexistent `setup.py`).

## [0.1.2] - 2026-01-07

### Added
- Full-text notes integration for semantic search.
- Extra citation key display support (Better BibTeX).

## [0.1.1] - 2025-12-29

### Added
- EPUB annotation support with CFI generation.
- Annotation feature documentation.
- Semantic search with ChromaDB and multiple embedding model support (default, OpenAI, Gemini).
- Smart update system with installation method detection.
- ChatGPT integration via SSE transport and tunneling.
- Cherry Studio and Chorus client configuration support.

## [0.1.0] - 2025-03-22

### Added
- Initial release.
- Zotero local and web API integration via pyzotero.
- MCP server with stdio transport.
- Claude Desktop auto-configuration (`zotero-mcp setup`).
- Search, metadata, full-text, collections, tags, and recent items tools.
- PDF annotation extraction with Better BibTeX support.
- Smithery and Docker support.
