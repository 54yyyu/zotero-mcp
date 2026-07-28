# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.4] - 2026-07-28

### Added
- **Listed in the official MCP Registry** (`registry.modelcontextprotocol.io`) as `io.github.54yyyu/zotero-mcp`. A `server.json` at the repo root describes the PyPI package, its stdio transport, and the four env vars a client needs to prompt for (`ZOTERO_LOCAL`, `ZOTERO_API_KEY`, `ZOTERO_LIBRARY_ID`, `ZOTERO_LIBRARY_TYPE`); ownership is proven by the `mcp-name:` marker in the README, which is what PyPI serves as the package description. The release workflow republishes on every tag via GitHub OIDC, so the registry version can't drift from PyPI. Registry clients install by the distribution name rather than the console-script name, so `zotero-mcp-server` is now also a console-script alias for `zotero-mcp` and `uvx zotero-mcp-server serve` works without `--from`.

### Fixed
- **`zotero-mcp update` no longer offers, or performs, a downgrade.** The check was `current_version != latest_version`, so any install ahead of the last PyPI release — every git checkout and dev build — was told an update was available, and running it replaced the newer code with the older release. Ordering is now compared rather than inequality, via `packaging.version.Version` where available and a numeric-tuple fallback where it isn't (`packaging` is not a declared dependency, only a common transitive one). Being ahead is reported as such instead of as a bare "already up to date", and `--force` from an ahead version warns before downgrading. As a side effect the comparison is numeric rather than lexical, so `0.10.0` is correctly newer than `0.9.0`.
- **Linked-file attachments are readable in local mode.** `download_attachment_file` started at `zot.dump()`, and for a *linked* attachment Zotero's local API answers `/file` with a 302 to a `file://` URL that httpx refuses to follow ("unsupported protocol"). A linked file is by definition never uploaded to WebDAV or Zotero storage either, so all three remaining sources failed on a file sitting readable on the same disk. The attachment path is now resolved out of `zotero.sqlite` first, which also spares ordinary stored files an API round trip. `zotero_get_annotations` was the visible casualty — unlike the fulltext and `read_pdf` paths it had no local-database step in front of the downloader — and its "linked attachments cannot be accessed remotely" error now points at `ZOTERO_LOCAL=true` instead, since that is the condition under which they do work. The resolved file is copied rather than handed back by path: callers treat the result as scratch and delete it, which on a linked file would have destroyed the user's original (the `_cleanup_path` failure mode from #372).

## [0.6.3] - 2026-07-28

### Added
- **`zotero_attach_file`** — attaches a local file or a direct PDF URL to an *existing* item, the gap that `zotero_add_from_file` (which always creates a new item) left open. Idempotent on both stored filename and content hash, so re-running converges instead of accumulating duplicate attachments, and the created attachment's key is returned so follow-up calls don't need a `get_item_children` round trip (#377, #386).
- **Multi-library tagging in the semantic search index (#163, phase 1)** — ChromaDB documents are now tagged with a `group_id` field (0 = personal library, else the Zotero groupID — the same `0`/groupID convention already used by `zotero_switch_library`). `zotero_semantic_search` gains a `library_id` parameter (accepting `0`, `"user"`, or a group's numeric groupID; omitted to search every indexed library — the new default) that filters on `group_id` DB-side via a ChromaDB `where` clause, never a Python post-filter. Existing collections are migrated automatically on the next `zotero_update_search_database` run (metadata-only backfill, no re-embedding, no downtime). Note: full item enrichment for a search hit still uses the currently active library's client, so a hit from a group library other than the active one may show limited detail until that capability lands in a follow-up; the incremental-update deletion pass is likewise not yet scoped per library (#404) (#396).
- **Multi-flavor Docker images published to GHCR** — `core` and `all` flavors built multi-arch on tags and `main`, with a container entrypoint supporting both MCP server mode and standalone CLI mode (`ZOTERO_APP=cli`). Documented container env vars and the ChromaDB/config persistence mount path in the README and `docs/docker-images.md` (#332).
- **Top-level `zotero_db_path` config setting**, with `semantic_search.zotero_db_path` still honored as a fallback. Config parsing moved to a typed, Pydantic-validated `zotero_mcp/config.py`: unknown keys are ignored, null nested objects fall back to defaults, and a malformed config logs a warning rather than crashing (#367).

### Fixed
- **Failed file uploads are no longer reported as successes.** pyzotero's `attachment_both()` signals a client-side rejection by returning the payload in its `failure` list rather than raising, and every attach site guarded only against exceptions — so a file that never landed was reported as "File attached" / "PDF attached". This was the shared root cause behind the silent-attach reports in #278, #306 and #399. Every attach path now verifies that an attachment was actually registered, retries through a create-then-upload flow that confirms the stored `md5`, and deletes the orphaned attachment shell if the upload still fails. Uploading through a local-only client now fails immediately with a clear message instead of a 404 from `localhost:23119`, which has no attachment endpoints (#403).
- **`children()` calls are paginated everywhere.** pyzotero's `children()` is an unpaginated passthrough, so without an explicit limit the Zotero API returns its default page of 25 and every call site silently operated on at most the first 25 children. Worst case was in `merge_duplicates`: only the first ≤25 children of each duplicate were re-parented onto the keeper and the rest went to the Trash with the duplicate item, in the operation users most expect to be conservative. Also fixed truncated results in semantic-search fulltext fetching, `get_attachment_details`, annotation extraction, item children listings and `zotero_library_coverage` (#387).
- **The semantic-search sync watermark is now per library.** A single `last_sync_version` scalar was compared against whichever library was active, so switching with `zotero_switch_library` and updating the database compared one library's version counter against another's and then overwrote it, corrupting sync state for both. Watermarks are stored per library under `last_sync_versions`, keyed by group_id. Existing configs migrate: the old scalar is kept for the env-configured default library (the only library it could have been tracking across restarts) and discarded if a runtime library override is active, since its provenance is then unknowable. A watermark ahead of the library's current version now triggers a full scan rather than a `since=` query that returns nothing (#393).
- **`zotero_get_pdf_outline` no longer takes the whole MCP server down.** `fitz.Document.get_toc()` segfaults on some born-digital journal PDFs; a segfault cannot be caught in-process, so the server died with it ("Server disconnected") and left orphaned processes behind. The outline is now read in a short-lived child process with a timeout — a crash or hang comes back as a plain error message and the server keeps running. The download also moved from the legacy `zot.dump()` call to the shared multi-source downloader, so WebDAV- and local-storage-backed attachments work, and a key naming the PDF attachment itself is accepted (#372).
- **`zotero_read_pdf_pages` accepts a PDF attachment key**, as its description always claimed. It previously scanned the key's children for a PDF — an attachment has none — and answered "No PDF attachment found" for the very item that *was* the PDF. The local reader gained `get_attachment_by_key()` for this, since attachments are invisible to the item-level lookup (#372).
- **Ollama semantic search works at query time.** zotero-mcp's `OllamaEmbeddingFunction` registers with ChromaDB under the name `ollama`, colliding with chromadb's own built-in of that name; when chromadb rebuilt the embedding function from the persisted collection config at query time, the built-in could win and reject our config with "This code should not be reached in query" — even though indexing had just succeeded. The persisted config now carries the keys both classes need, `build_from_config` accepts either spelling, and our classes are re-registered immediately before a collection is opened so a late import can't shadow them. The same treatment was applied to the `openai` and `huggingface` registrations (#382).
- **`zotero_export_bibliography` works in local and hybrid mode.** Rendering was routed to whichever client was active, and Zotero's local API has no CSL engine, so it failed with "Local API does not support Atom output". Bibliography and citation rendering now goes through the web API whenever credentials are configured (with the active library override applied, so a switched-to group library is targeted), and local-only mode returns an actionable message instead (#371).
- **Annotation tags are returned by `zotero_get_annotations`.** Two extraction paths hardcoded an empty tag list, and the Better BibTeX path dropped tags one layer further down in `process_annotation()`. Tag shapes from all sources are normalized, so a loose shape can no longer turn into an error (#377).
- **`db-inspect --stats` and `--filter` work on very large collections.** `--stats` issued one unbounded `col.get()`, which exceeds SQLite's bound-variable ceiling past a few tens of thousands of rows ("too many SQL variables"). `--filter` applied `--limit` to the raw fetch instead of to matches, so it only ever scanned the first N records in storage order and reported "No records matched your filter" for terms that were plainly present. Both now stream the collection in fixed-size batches with flat memory use (#368, #384).
- **Setup writes the Claude Desktop config where Claude Desktop actually reads it.** Some builds keep it in `Claude-3p` rather than `Claude` (`%LOCALAPPDATA%` on Windows, and the same directory name on macOS and Linux), so setup reported success against a file the running app never read. Every known location is probed, all existing configs are written rather than just the first match, and the absolute path(s) written are always printed. Reading the config back for environment variables had the same first-match-wins problem and now scans all of them (#392).
- **Attachments upload to WebDAV directly when WebDAV is configured**, instead of being pushed to Zotero cloud storage first — which is the wrong target for those users and fails with HTTP 413 once the free 300 MB quota is full. If the WebDAV PUT fails after the attachment item was created, the orphaned item is deleted rather than left behind with no file bytes (#361).
- **`zotero_db_path` is honored by every local-mode tool.** `list_libraries`, `validate_library_switch`, `list_feeds`, `get_feed_items`, `read_pdf`, `get_item_fulltext` and `get_attachment_path` ignored the configured path and relied on auto-detection, so they read the wrong database for anyone with a non-default Zotero profile location (#367).
- **The BibTeX fallback emits ISBN and ISSN.** When Better BibTeX isn't reachable, the hand-rolled builder omitted both, so books and journal articles lost identifiers the underlying record carried (#398).
- **`zotero_read_pdf_pages` no longer risks deleting files it doesn't own.** Its cleanup step removes the *parent directory* of the PDF it worked on, guarded only by a `startswith(tempfile.gettempdir())` check. That guard is too loose in both directions: on Linux `gettempdir()` is `/tmp`, so a path like `/tmp/paper.pdf` has `/tmp` as its parent and passes, and the tool also ran cleanup on files resolved out of the user's own Zotero storage. Cleanup is now restricted to directories this tool created (`zotero_pdf_` prefix, strictly below the temp root), and a file resolved from local storage is never passed to it. Found because it wiped the CI runner's temp directory mid-suite, which had also been misattributed to an unrelated PR.
- **Install hints match how zotero-mcp was actually installed.** Runtime "install the extra" errors hardcoded a `pip install` command, which silently does nothing useful for `uv tool` and pipx installs. The flavor is detected from the package's own path and only working commands are shown; when it can't be determined, all three are listed (#388).

## [0.6.2] - 2026-07-13

### Added
- **Opt-in collection filter for the semantic search database** — set `semantic_search.collection_keys` in `config.json` to build the vector database from only those collections (subcollections included, resolved recursively) instead of the whole library. Unset, behavior is unchanged (#370).

### Fixed
- **Chunking no longer forces a full re-extract and re-embed of the whole library on every update.** With `semantic_search.chunking` enabled (#350), items are indexed only under their chunk ids (`<key>#<n>`), but the "already indexed?" check in the local-fulltext path looked each item up by its bare key. That exact-id lookup never matched, so every item counted as new: each fulltext update re-extracted every PDF and re-embedded the entire library, silently and without an error. `get_document_metadata` now falls back to chunk 0, which carries the item-level `date_modified` and `has_fulltext` that the check needs, so unchanged items are skipped again (#380).
- **Local database auto-discovery now honors a custom Zotero data directory** — the `extensions.zotero.dataDir` preference is read from the Zotero profile's `prefs.js` (macOS/Windows/Linux), so relocated data directories no longer fail with "Zotero database not found at ~/Zotero/zotero.sqlite" (#68). The `ZOTERO_DB_PATH` environment variable, documented in the README but previously unimplemented, now works as an override, and the not-found error lists every location checked plus how to fix it. `extensions.zotero.baseAttachmentPath` is likewise read from the profile's `prefs.js`, fixing resolution of linked attachments relative to a base directory (#379).
- **Items whose full-text extraction once failed are retried when their attachments change** — attaching a PDF later doesn't bump the parent's `dateModified`, so items first indexed metadata-only were permanently locked out of full-text indexing. The local-mode scan now records each item's attachment-key set and retries when it changes; legacy "failed" records retry once and converge (#373).
- **ChromaDB upserts are split to the backend's max batch size** — with chunking enabled, a batch of long documents could exceed ChromaDB's `max_batch_size` (~5,461), failing the whole batch and degrading the retry pass to one-record-at-a-time upserts (#369).
- **Created annotations land in the right place on PDFs with a non-zero page box origin** — highlight rectangles were written in PyMuPDF's CropBox-normalized space, but Zotero positions annotations in native PDF user space (MediaBox origin). Rects (and the derived sort index) are now mapped through the page's inverse transformation matrix, which also handles page rotation (#381).

## [0.6.1] - 2026-07-03

### Fixed
- **`zotero_get_search_database_status` no longer reports "0 documents / not initialized" against a populated database** — ChromaDB ≥1.x's embedding-function conflict check rejected the status reader's no-op embedding function; it now identifies as `"default"`, which short-circuits the check for any persisted backend (#362, #364).
- **Semantic search with the reranker enabled no longer times out** — the cross-encoder was reloaded from disk on every request (~30s per call); it is now cached process-wide and warmed up in the background at server start, so reranked searches are sub-second after the first load (#283, #365).
- **Ollama embeddings now use the current `/api/embed` endpoint** instead of the deprecated `/api/embeddings` route. The whole batch is sent in a single request (`input`) rather than one request per document, and the response's `embeddings` list is parsed accordingly (#349, #360).

## [0.6.0] - 2026-06-22

### Added
- **Passage-level chunking for semantic search** (opt-in via `semantic_search.chunking`) — each item is indexed as overlapping passages with char/page provenance, so search returns a grounded snippet and long PDFs stay searchable past the single-vector truncation limit. Off by default; enabling it needs a one-time `update-db --force-rebuild` (#350).
- **Agentic research tools** (#350):
  - `zotero_find_related_papers` — walks the OpenAlex citation graph (references + citing works) and flags each result as already-in-library or a gap.
  - `zotero_library_coverage` — audits which items lack a PDF, with DOIs ready for the OA download cascade.
  - `zotero_synthesize_annotations` — per-paper digest of highlights and notes.
  - `zotero_export_bibliography` — CSL-rendered bibliography / citations / BibTeX via Zotero's own engine.
- **MCP prompts and resources** — `literature_review`, `synthesize_my_notes`, `find_contradicting_evidence`, `expand_from_paper`; resources `zotero://collections`, `zotero://items/{key}`, `zotero://collections/{key}/items` (#350).
- **`zotero_batch_update_extra`** — batch upsert/remove of `Key: value` lines in the Extra field across many items (#232, #334).
- **Collection resolution in all add paths** — collection specs (key, name, or parent/child path) are resolved and validated across every add path and `manage_collections`; an unknown or ambiguous spec fails the add early with suggestions instead of leaving an unfiled item (#336, #340).
- **Idempotent adds** — `if_exists=duplicate|file|skip`: re-adding converges (files into missing collections, adds missing tags) instead of duplicating. MCP default stays `duplicate`; the CLI defaults to `file` (#337, #341).
- **`zotero-cli add isbn|bibtex|csl-json` subcommands**, with stdin via `-` (#338, #342).
- **Ollama embedding backend** for semantic search (`nomic-embed-text`, `bge-m3`) (#349).
- **OpenAI Batch API embedding indexing** — submit / status / import async embedding jobs for cheaper large-library indexing (#346).
- **OpenAI embedding sub-batching and rate limiting** — `embedding_config.request_batch_size` (default 64, for stricter OpenAI-compatible providers) and an optional `embedding_config.rate_limit_rps` per-request throttle for 429 safety (#261, #307, #356).
- **`citation_key` on `zotero_update_item`** — writes the native `citationKey` field (#320, #321).
- **`ZOTERO_WEBDAV_TIMEOUT`** env var to tune the WebDAV upload read timeout (#344, #345).
- Standalone PDF attachments now surface in `zotero_get_collection_items` (#224).

### Fixed
- Incremental semantic sync no longer advances the watermark when the immutable sqlite snapshot lags the live API (un-checkpointed WAL), which previously made newly-added items be skipped permanently (#292, #333).
- `update-db --fulltext` no longer caps each item at a single truncated vector; passage chunking indexes full text past the embedding limit (#290).
- `zotero-cli add file` no longer raises `TypeError` from a phantom `parent_key`; exposes `--title` / `--item-type` (#335, #339).
- `zotero_read_pdf_pages` routes through the shared multi-source download (local → WebDAV → cloud), so WebDAV-backed PDFs work (#351).
- Scite reaches the `/papers` endpoint correctly — it now sends a bare JSON array instead of a `{"dois": [...]}` object (which returned HTTP 400 and broke retraction checks), and matches Scite's lowercased DOI keys so uppercase DOIs (e.g. `10.1016/S0140-6736(97)11096-0`) aren't missed (#331).
- Semantic search reliability: deterministic embeddings via explicit `encoding_format="float"` (fixes intermittent OpenRouter/Gemini "No embedding data received"); `db-status` no longer loads an embedding model or holds the global API lock (#348).

## [0.5.0] - 2026-06-08

### Added
- **`zotero_get_page_layout` tool** — detect figure/table regions on a PDF page with bounding boxes and caption association, for coordinate-grounded reading (#312).
- **`zotero_add_by_bibtex`** — ingest one or more items from a BibTeX string OR a `.bib`/`.bibtex` file path; parses via `bibtexparser` (with LaTeX→unicode conversion), maps to Zotero item format, preserves the citation key in Extra, and attempts an open-access PDF attachment when a DOI is present (#241).
- **`zotero_add_by_csl_json`** — same for CSL JSON input from an inline string/object/array OR a `.json`/`.csljson` file path. The CSL `id` is preserved in Extra as the citation key (#241).
- New `citation_import` module — BibTeX parsing, CSL JSON coercion, and the shared field/type crosswalk (reference: <https://aurimasv.github.io/z2csl/typeMap.xml>).
- **`zotero_read_pdf_pages` tool** — read a specific page range from a PDF attachment after section identification via `zotero_get_pdf_outline`. Extracts text from the requested pages using PyMuPDF, avoiding the need to read the entire paper when only a few pages are relevant.
- RSS feed items now surface their publication date (and DOI) (#316).

### Changed
- Bumped the `pyzotero` floor to `>=1.8.0` — the first release accepting the custom HTTP/1.1 `client=` used by the local-API fix; older `1.6.x`/`1.7.x` crashed every tool call with `TypeError: unexpected keyword argument 'client'` (#322).
- Bumped the `[semantic]` extra's `chromadb` floor to `>=1.0.0` for `register_embedding_function`, introduced in chromadb 1.0.0 (#324).
- New base dependency: `bibtexparser>=1.4,<2`.

### Fixed
- `zotero_search_by_citation_key` now matches the native `citationKey` field, not just the `Extra` fallback (#319).
- Custom OpenAI/Gemini/HuggingFace embedding functions are registered with ChromaDB's registry so a persisted database reloads correctly (#315).
- Bounded the global Zotero API lock so a stuck operation can't wedge every tool with opaque `-32001` timeouts (#311).
- `zotero_add_by_url` arXiv path is resilient to arXiv outages via a CrossRef fallback (#310).
- `zotero_add_by_doi` and arXiv PDF uploads now honor `ZOTERO_WEBDAV_*` instead of always going to Zotero cloud storage (#314, #313).
- Strip the pyzotero-rejected `lastRead` field on attachment updates, fixing `zotero_update_item` failures on attachments opened in Zotero's PDF reader (#318, #317).

### Security
- **SSRF guard on the open-access PDF download path** — the OA-PDF URL comes from third-party metadata APIs (Unpaywall / Semantic Scholar) and was previously fetched with no scheme/host validation and default redirect-following. It is now validated against a public-host allowlist (rejecting loopback / link-local / RFC1918 / cloud-metadata) with per-redirect-hop re-checking (#327, #326).
- **Credential-hygiene + DoS hardening**: mask `ZOTERO_API_KEY` in `setup --no-claude` output by default (`--show-secrets` to opt in); write credential config files with `0o600`; prefer the env var / `getpass` over the `--api-key` flag; add a subprocess timeout to `pdfannots2json`; run the Docker image as a non-root user (#328, #326).

## [0.2.2] - 2026-03-26

### Added
- **Scite citation intelligence integration** — the MCP counterpart of the [Scite Zotero Plugin](https://github.com/scitedotai/scite-zotero-plugin). New optional `[scite]` extra that enriches Zotero library items with citation data from [scite.ai](https://scite.ai). No Scite account required (#180).
  - `scite_enrich_item`: Get citation tallies (supporting/contrasting/mentioning) and editorial notice alerts for any paper by DOI or Zotero item key.
  - `scite_enrich_search`: Search your Zotero library and see Scite tallies and retraction alerts inline with each result.
  - `scite_check_retractions`: Scan your library (by collection, tag, or recent items) for retractions, corrections, and other editorial notices.
- New `scite_client.py` module: thin HTTP client for `api.scite.ai` public endpoints (tallies, paper metadata, editorial notices).

### Fixed
- **macOS PDF extraction deadlock** — replaced `multiprocessing.Process` with `subprocess.run` to prevent FastMCP re-initialization in child process (#178, #173, #181).
- **Deleted items indexed in semantic search** — excluded trashed items from `get_items_with_text()` and `get_item_count()` (#175).

## [0.2.1] - 2026-03-22

### Fixed
- **`create_annotation` crash** — fixed `_client._client.` double-indirection typo introduced in v0.2.0 refactor (#168).
- **`attachments:` path resolution** — now reads `baseAttachmentPath` from Zotero's `prefs.js` instead of wrongly resolving against the storage directory (#169).

## [0.2.0] - 2026-03-22

### Architecture
- **Split `server.py` (4,800 lines) into `tools/` subpackage** — search, retrieval, annotations, write, connectors, and shared helpers are now separate modules. `server.py` is a 109-line re-export shim.
- **Removed `_ServerModule` sys.modules hack** — tool modules use module-level attribute access; tests patch canonical locations directly.
- **Optional dependency groups** — `[semantic]` (ChromaDB, embeddings), `[pdf]` (PyMuPDF, EPUB), `[all]`. Base install is lightweight with no ML dependencies.

### Refactored
- Deduplicated 7 item-formatting functions into single `format_item_result()` with configurable abstract length, tags, and extra fields.
- Extracted `_normalize_limit()` helper replacing 12 copy-pasted `isinstance(limit, str)` blocks.
- Consolidated duplicate `suppress_stdout()` into `utils.py`.
- Merged `_strip_xml_tags()` into `clean_html()` with `collapse_whitespace` parameter.
- Extended `format_creators()` to handle string creators; `_format_bbt_result()` now delegates to it.
- Collapsed `get_annotations`/`_get_annotations` wrapper into single function.
- Modernized typing in 5 modules: `Optional[X]` → `X | None`, `Dict` → `dict`, `List` → `list`.
- Removed dead code: unused `_extract_item_key_from_input()` function, stale typing imports across 7 modules.

### Fixed
- **Stale embedding model detection** — ChromaDB collections created with a deprecated model (e.g., `text-embedding-004`) are now auto-detected and recreated on startup.
- **Bare `except:` clauses** — replaced with specific exception types in `better_bibtex_client.py`.
- **PDF outline import order** — defers PyMuPDF import until after attachment check.
- **Suppressed noisy pdfminer warnings** during PDF text extraction.

### Docs
- README documents optional extras (`[semantic]`, `[pdf]`, `[all]`), write operations, and embedding model troubleshooting.
- Removed stale fork enhancements section.

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
