# Tools

Every tool the MCP server provides, and how to choose which ones it exposes.

<a id="tool-groups"></a>

## Tool groups

Every tool this server registers is sent to the model on **every** request, so
the tool list is a fixed tax on your context window before you type anything.
To keep that cost proportionate, optional capabilities are grouped into
*toolsets* that you turn on when you need them.

Set `ZOTERO_MCP_TOOLSETS` to control which groups are exposed:

| Value | Effect |
|---|---|
| *(unset)* | Default profile — core tools plus `libraries`, `search-admin`, `pdf-geometry` |
| `all` | Everything (the pre-0.9 behaviour) |
| `none` | Core tools only — the smallest surface |
| `scite,feeds` | Core plus the named groups |
| `all,-scite` | Everything except the named groups |

Values are case-insensitive and may be comma- or space-separated. An unknown
group name is an error at startup rather than a silent no-op.

| Group | Default | Contents |
|---|---|---|
| `scite` | off | Scite citation tallies and retraction checks (calls scite.ai; pairs with the `[scite]` extra) |
| `duplicates` | off | Find and merge duplicate items — library maintenance |
| `discovery` | off | `find_related_papers`, `library_coverage` — corpus-level exploration |
| `feeds` | off | Zotero RSS feed subscriptions |
| `relations` | off | Explicit item-to-item "related items" links |
| `libraries` | **on** | List and switch between personal/group libraries |
| `search-admin` | **on** | Build and inspect the semantic search index |
| `pdf-geometry` | **on** | Page layout and PDF outline — pairs with area annotations |
| `chatgpt-connector` | auto | The `search`/`fetch` pair required by ChatGPT deep research |

`chatgpt-connector` is scoped by transport: it turns on automatically when the
server is served over `streamable-http` or `sse` (how ChatGPT reaches it) and
stays off for `stdio`. Name it explicitly to override either way.

Anything not listed above is **core** and always available.

**Note:** a disabled tool is genuinely absent — not merely hidden — so the
model cannot call it. If you rely on a capability, enable its group.

Example (Claude Desktop / Claude Code):

```json
"env": {
  "ZOTERO_LOCAL": "true",
  "ZOTERO_MCP_TOOLSETS": "scite,duplicates"
}
```

## Available tools

> Availability depends on your `ZOTERO_MCP_TOOLSETS` setting — see
> [Tool groups](#tool-groups) above.

### 🧠 Semantic search tools
- `zotero_semantic_search`: AI-powered similarity search with embedding models
- `zotero_update_search_database`: Manually update the semantic search database
- `zotero_get_search_database_status`: Check database status and configuration

### 🔍 Search tools
- `zotero_search_items`: Search your library by keywords
- `zotero_advanced_search`: Perform complex searches with multiple criteria
- `zotero_get_collections`: List collections
- `zotero_get_collection_items`: Get items in a collection
- `zotero_get_tags`: List all tags
- `zotero_get_recent`: Get recently added items
- `zotero_search_by_tag`: Search your library using custom tag filters

### 📚 Content tools
- `zotero_get_item_metadata`: Get detailed metadata (supports `format="markdown"`, `format="json"` for complete raw Zotero metadata, and `format="bibtex"`)
- `zotero_get_item_fulltext`: Get full text content
- `zotero_get_item_children`: Get attachments and notes for one item or many (pass an array of keys)
- `zotero_read_pdf_pages`: Read a page range of a PDF as Markdown, with pages whose math, figures or tables the text garbles flagged; `format="image"` returns the pages as images, and `rect=[x, y, width, height]` zooms into one region
- `zotero_get_pdf_outline`: Extract the table of contents / outline from a PDF attachment

### 📝 Annotation & notes tools
- `zotero_get_annotations`: Get annotations (including direct PDF extraction); use `format="json"` for normalized records suitable for scripts and other MCP tools
- `zotero_synthesize_annotations`: Build a per-paper annotation/note digest; supports `format="json"` for structured grouped output
- `zotero_get_notes`: Retrieve notes from your Zotero library; pass `query` to search note and annotation text instead of listing
- `zotero_create_annotation`: Create a highlight (`text=`) or an area annotation (`rect=[x, y, width, height]`)
- `zotero_update_annotation`: Change an annotation's text, comment, color or tags
- `zotero_delete_annotation`: Permanently delete an annotation
- `zotero_manage_note`: Create, update, or delete a note via `action="create"|"update"|"delete"` (beta feature)
- `zotero_get_page_layout`: Detect figure, table and display-equation regions on a PDF page (with captions or equation numbers and normalized coordinates) for accurate area annotation placement — its reported `bbox` can be passed straight to `zotero_create_annotation(rect=...)`

### 📊 Scite citation intelligence tools

> Opt-in group: enable with `ZOTERO_MCP_TOOLSETS=scite` — see [Tool groups](#tool-groups).

- `scite_enrich_item`: Get Scite citation tallies and retraction alerts for a paper — the MCP version of the [Scite Zotero Plugin](https://github.com/scitedotai/scite-zotero-plugin)
- `scite_enrich_search`: Search your Zotero library with Scite-enriched results (tallies + alerts inline)
- `scite_check_retractions`: Scan items for retractions and editorial notices

No Scite account is required; these use public API endpoints.

### ✏️ Write access tools
- `zotero_authorize_local_writes`: Request local write permission from Zotero (Zotero 10+) — opens a dialog in the Zotero app
- `zotero_write_capabilities`: Report which write path is available (local / hybrid / web / none) and what to do about it

### 📦 Item & collection management tools
- `zotero_add_by_doi`: Add a paper by DOI with automatic metadata and open-access PDF attachment (Unpaywall, arXiv, Semantic Scholar, PMC)
- `zotero_add_by_url`: Add a paper by URL (arXiv, DOI URLs, and general webpages)
- `zotero_add_by_isbn`: Add a book by ISBN (Open Library + Google Books cascade)
- `zotero_add_by_bibtex`: Add one or more items from BibTeX (inline or .bib file)
- `zotero_add_by_csl_json`: Add one or more items from CSL JSON (inline or file)
- `zotero_add_from_file`: Import a local PDF or EPUB file with automatic DOI extraction

All add tools take a `collections` parameter accepting collection keys, names, or `parent/child` paths — resolved and validated before the item is created, so unknown or ambiguous specs fail with suggestions instead of producing an unfiled item. They also take `if_exists` (`"duplicate"` — default — always creates; `"file"` reuses an existing item matching the DOI/arXiv ID/ISBN/URL, filing it into missing collections and adding missing tags; `"skip"` leaves a match untouched) and `create_missing_collections` (create unknown collection specs, including path chains, instead of failing). The `zotero-cli add` commands default to `--if-exists file`.

- `zotero_attach_file`: Attach a local file or a PDF URL to an existing item by key (no new item created; returns the attachment key; idempotent per filename and content hash)
- `zotero_set_item_parent`: Set, change, or clear an item's parent (`parent_key=null` makes it top-level)
- `zotero_create_collection`: Create a new collection (folder/project) in your library
- `zotero_update_collection`: Rename a collection or move it under another parent (keeps its key, subcollections and items)
- `zotero_search_collections`: Search for collections by name to find their keys
- `zotero_manage_collections`: Add or remove items from collections (accepts keys, names, or `parent/child` paths)
- `zotero_update_item`: Update metadata for an existing item (title, tags, abstract, date, etc.)
- `zotero_find_duplicates`: Find duplicate items by title and/or DOI, paged with `limit`/`offset`
- `zotero_merge_duplicates`: Merge duplicate items with dry-run preview; consolidates all child items. `auto=True` merges every high-confidence (same-DOI) group in one pass behind a two-call plan/confirm gate
- `zotero_search_by_citation_key`: Look up items by BetterBibTeX citation key (with Extra field fallback)

### 🔗 Related items tools
- `zotero_get_item_related`: Get all related items for a specific Zotero item
- `zotero_add_item_relation`: Add a related item relationship (creates bidirectional link)
- `zotero_remove_item_relation`: Remove a related item relationship

## Managing related items

Zotero MCP supports managing relationships between items in your library. This is useful for linking related papers, tracking versions, or connecting preprints to their published versions.

> These tools are in the opt-in `relations` group. Enable them with
> `ZOTERO_MCP_TOOLSETS=relations` — see [Tool groups](#tool-groups).

### View related items
```
zotero_get_item_related(item_key="ABCD1234")
```

### Add a relation
Create a bidirectional link between two items:
```
zotero_add_item_relation(
    item_key="ABCD1234",
    related_item_key="EFGH5678",
    relation_type="dc:relation"  # Optional, defaults to "dc:relation"
)
```

### Remove a relation
```
zotero_remove_item_relation(
    item_key="ABCD1234",
    related_item_key="EFGH5678",
    remove_bidirectional=True  # Also remove the reverse relation (default: true)
)
```

**Relation types:**
- `dc:relation` — General related items (default)
- `owl:sameAs` — Items that are the same work (e.g., preprint and published version)

## PDF annotation extraction

Zotero MCP includes advanced PDF annotation extraction capabilities:

- **Direct PDF processing**: Extract annotations directly from PDF files, even if they're not yet indexed by Zotero
- **Enhanced search**: Search through PDF annotations and comments
- **Image annotation support**: Extract image annotations from PDFs
- **Seamless integration**: Works alongside Zotero's native annotation system

For optimal annotation extraction, it is **highly recommended** to install the [Better BibTeX plugin](https://retorque.re/zotero-better-bibtex/installation/) for Zotero. The annotation-related functions have been primarily tested with this plugin and provide enhanced functionality when it's available.

The first time you use PDF annotation features, the necessary tools will be automatically downloaded.
