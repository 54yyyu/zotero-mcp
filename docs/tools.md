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
| *(unset)* | Default profile — core tools plus `semantic`, `libraries`, `search-admin`, `pdf-geometry` |
| `all` | Every group except `legacy` |
| `none` | Core tools only, the smallest surface (no semantic search; `none,semantic` adds it back) |
| `scite,feeds` | Core plus the named groups (the default groups are not included) |
| `default,scite` | The default profile plus the named groups |
| `all,-scite` | Everything except the named groups |

Values are case-insensitive and may be comma- or space-separated. An unknown
group name is an error at startup rather than a silent no-op.

| Group | Default | Contents |
|---|---|---|
| `scite` | off | Scite citation tallies and retraction checks (calls scite.ai; pairs with the `[scite]` extra) |
| `duplicates` | off | Find and merge duplicate items — library maintenance |
| `semantic` | **on** | `zotero_semantic_search` (needs the `[semantic]` extra) |
| `discovery` | off | `find_related_papers`, `library_coverage` — corpus-level exploration via OpenAlex |
| `feeds` | off | Zotero RSS feed subscriptions |
| `relations` | off | Explicit item-to-item "related items" links |
| `libraries` | **on** | List and switch between personal/group libraries |
| `search-admin` | **on** | Build and inspect the semantic search index (needs the `[semantic]` extra) |
| `pdf-geometry` | **on** | Page layout and PDF outline — pairs with area annotations |
| `chatgpt-connector` | auto | The `search`/`fetch` pair required by ChatGPT deep research |
| `legacy` | off | Pre-0.16 names of merged tools: `zotero_search_by_tag`, `zotero_create_collection`, `zotero_update_collection`, `zotero_delete_collection`. Removed in 0.17 |

`chatgpt-connector` is scoped by transport: it turns on automatically when the
server is served over `streamable-http` or `sse` (how ChatGPT reaches it) and
stays off for `stdio`. Name it explicitly to override either way.

`semantic` and `search-admin` need the `[semantic]` extra. On an install
without it they are left out automatically (logged at startup), unless you
name them in `ZOTERO_MCP_TOOLSETS`, in which case they stay and answer with
install instructions. A `ZOTERO_MCP_TOOLSETS` that uses `none` without naming
`semantic` logs one warning at startup, since `none` included semantic search
before; `none,-semantic` leaves it out without the warning.

`legacy` is enabled only by name, never by `all`: the merged tools already
cover every call. Use `ZOTERO_MCP_TOOLSETS=default,legacy` if a prompt or
config of yours still names an old tool, and move it to the new name before
0.17:

| Old tool | Use instead |
|---|---|
| `zotero_search_by_tag(tag=...)` | `zotero_search_items(tag=...)` with no `query` |
| `zotero_create_collection(name=...)` | `zotero_manage_collection(action="create", name=...)` |
| `zotero_update_collection(collection_key=..., ...)` | `zotero_manage_collection(action="update", collection_key=..., ...)` |
| `zotero_delete_collection(collection_key=...)` | `zotero_manage_collection(action="delete", collection_key=...)` |

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

> Availability depends on your `ZOTERO_MCP_TOOLSETS` setting; see
> [Tool groups](#tool-groups) above.

### 🔍 Search tools
- `zotero_search_items`: Keyword search over titles, creators and years, with tag conditions (`tag=["a OR b", "-c"]`); with only `tag`, lists items by tag
- `zotero_advanced_search`: Structured field conditions (date ranges, item types, one-field matches) joined by AND or OR
- `zotero_search_by_citation_key`: Look up an item by its BetterBibTeX citation key
- `zotero_get_recent`: Most recently added items
- `zotero_get_tags`: List all tags
- `zotero_get_collections`: List collections as a tree
- `zotero_get_collection_items`: Items in one collection, paged
- `zotero_search_collections`: Find collections by name to get their keys

### 🧠 Semantic search tools (`semantic`, `search-admin`)
- `zotero_semantic_search`: Find papers by meaning with embeddings
- `zotero_update_search_database`: Build or refresh the index
- `zotero_get_search_database_status`: Index state and configuration

### 📚 Content tools
- `zotero_get_item_metadata`: Metadata for one item (`format="markdown"`, `"json"` for the raw record, or `"bibtex"`)
- `zotero_get_item_fulltext`: Full text of an item's PDF or EPUB
- `zotero_get_item_children`: Attachments and notes for one item or many (pass an array of keys)
- `zotero_read_pdf_pages`: Read a page range of a PDF as Markdown, with pages whose math, figures or tables the text garbles flagged; `format="image"` returns the pages as images, and `rect=[x, y, width, height]` zooms into one region
- `zotero_get_pdf_outline`: A PDF's table of contents (`pdf-geometry`)
- `zotero_get_attachment_path`: Local file paths of an item's attachments (local mode)
- `zotero_export_bibliography`: Format references or in-text citations with Zotero's CSL engine, or export BibTeX

### 📝 Annotation & notes tools
- `zotero_get_annotations`: Get annotations (including direct PDF extraction); `format="json"` returns normalized records
- `zotero_synthesize_annotations`: Per-paper digest of highlights and notes across a collection or tag
- `zotero_get_notes`: List notes, or pass `query` to search note and annotation text
- `zotero_manage_note`: Create, update, or trash a note via `action="create"|"update"|"delete"`
- `zotero_create_annotation`: Highlight (`text=`), area box (`rect=[x, y, width, height]`) or sticky note (`note=[x, y]`)
- `zotero_update_annotation`: Change an annotation's text, comment, color or tags
- `zotero_delete_annotation`: Permanently delete an annotation
- `zotero_get_page_layout`: Detect figure and table regions on a PDF page with normalized boxes, for area annotations (`pdf-geometry`)

### 📦 Adding and editing items
- `zotero_add_item`: Add items from a DOI, URL, ISBN, PMID, BibTeX, CSL JSON or a local file. Takes `collections` (keys, names or `parent/child` paths, validated before anything is created), `if_exists` (`"duplicate"`, the default, always creates; `"file"` reuses an item matching the DOI/ISBN/URL and files it into missing collections; `"skip"` leaves a match untouched) and `create_missing_collections`. The `zotero-cli add` commands default to `--if-exists file`.
- `zotero_attach_file`: Attach a local file or a PDF URL to an existing item (returns the attachment key; skips a file already attached)
- `zotero_update_item`: Update metadata, tags and collection membership of one item
- `zotero_batch_update`: Add or remove tags and Extra lines across many items
- `zotero_set_item_parent`: Set, change, or clear an item's parent (`parent_key=null` makes it top-level)
- `zotero_delete_item`: Move an item to the Trash

### 🗂 Collections
- `zotero_manage_collection`: Create, rename or move, or delete a collection via `action="create"|"update"|"delete"`
- `zotero_set_item_collections`: Add items to or remove them from collections (keys, names, or `parent/child` paths)

### ✏️ Write access tools
- `zotero_authorize_local_writes`: Request local write permission from Zotero (Zotero 10+); opens a dialog in the Zotero app
- `zotero_write_capabilities`: Report which write path is available (local / hybrid / web / none) and what to do about it

### 🏛 Libraries (`libraries`)
- `zotero_list_libraries`: Personal, group and feed libraries with their IDs
- `zotero_switch_library`: Set the library every later call works on

### 🧹 Duplicates (`duplicates`)
- `zotero_find_duplicates`: Find duplicate items by title and/or DOI, paged with `limit`/`offset`
- `zotero_merge_duplicates`: Merge duplicates with a dry-run preview; `auto=True` merges every same-DOI group behind a two-call plan/confirm gate

### 🌐 Discovery (`discovery`) and feeds (`feeds`)
- `zotero_find_related_papers`, `zotero_library_coverage`: Related papers and field coverage via OpenAlex
- `zotero_list_feeds`, `zotero_get_feed_items`: Zotero RSS feed subscriptions

### 📊 Scite citation intelligence tools (`scite`)
- `scite_enrich_item`: Scite citation tallies and retraction alerts for a paper, the MCP version of the [Scite Zotero Plugin](https://github.com/scitedotai/scite-zotero-plugin)
- `scite_enrich_search`: Search your library with Scite-enriched results
- `scite_check_retractions`: Scan items for retractions and editorial notices

No Scite account is required; these use public API endpoints.

### 🔗 Related items tools (`relations`)
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
