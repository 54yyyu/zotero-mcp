# Configuration

How Zotero MCP connects to your library, and every setting that changes its behaviour. For first-time setup of a specific client, start with [Getting started](getting-started.md).

## Connection modes

- **Local read-only**: offline access to a running Zotero, no credentials at all (`ZOTERO_LOCAL=true`)
- **Local read-write** (Zotero 10+): add one authorization and writes stay local too. See [Local write support](#local-write-support)
- **Web API**: cloud library access from anywhere (`ZOTERO_API_KEY` + `ZOTERO_LIBRARY_ID`)
- **Hybrid**: read from local Zotero, write via the web API (`ZOTERO_LOCAL=true` plus web credentials), for anyone on an older Zotero

### Using the web API instead of the local API

For accessing your Zotero library via the web API (useful for remote setups):

```bash
zotero-mcp setup --no-local --api-key YOUR_API_KEY --library-id YOUR_LIBRARY_ID
```

Generate an API key at [zotero.org/settings/security](https://www.zotero.org/settings/security#applications). `ZOTERO_LIBRARY_ID` is your numeric **userID**, shown on that same page (for a group library, use the group's ID and also set `ZOTERO_LIBRARY_TYPE=group`).

<a id="local-write-support"></a>

## Local write support (Zotero 10+)

Zotero's local API was read-only for most of its life, which is why writes have always
gone through the Zotero web API. Zotero 10 added local write endpoints, so on that
version writes can stay entirely on your machine: no cloud account, no waiting for a
sync round-trip, and no Zotero Storage quota to run into when attaching PDFs.

Writes need a **local API key**. It has nothing to do with a zotero.org API key and
can't be created in advance — Zotero grants it through a dialog:

```bash
zotero-mcp authorize-local
```

Zotero shows a prompt with three buttons. **Always Allow** grants a reusable key that
is saved to `~/.config/zotero-mcp/config.json` (owner-only permissions) and picked up
automatically from then on — that's the one you want. **Allow** grants a key valid for
exactly one write; it is never saved to disk, since a consumed key sitting there would
take priority over working web credentials on every later run. **Deny** grants nothing.

If a key is ever rejected — consumed, revoked from Zotero's side, or belonging to a
Zotero database you no longer have — the server drops it and falls back to web
credentials if you have them, rather than failing against a key that cannot work again.

An MCP client can do the same thing without dropping to a shell: the
`zotero_authorize_local_writes` tool opens the same dialog, and
`zotero_write_capabilities` reports which write path is currently available. If a write
is refused, that second tool tells you whether the fix is authorizing or supplying web
credentials.

```bash
zotero-mcp authorize-local --status    # what can write right now
zotero-mcp authorize-local --revoke    # forget the stored key
zotero-mcp authorize-local --print     # print it for ZOTERO_LOCAL_API_KEY instead
```

Nothing here is required. Without a local key the server behaves exactly as before:
hybrid mode if web credentials are set, and a message explaining both options if not.
Set `ZOTERO_LOCAL_WRITE=false` to keep writes on the web API even when a key exists.

**Revoking.** `--revoke` removes the copy this server stores. Zotero keeps its own
record of what it has authorized, cleared from **Settings → Advanced → Clear Write
Authorizations**.

**Requirements.** Zotero 10 or newer, with "Allow other applications on this computer
to communicate with Zotero" enabled in Settings → Advanced. Older versions have no
local write endpoints at all; `zotero-mcp authorize-local` detects this and says so
rather than opening a dialog that can't help.

## Environment variables

**Zotero connection:**
- `ZOTERO_LOCAL=true`: Use the local Zotero API (default: false)
- `ZOTERO_API_KEY`: Your Zotero API key (for web API)
- `ZOTERO_LIBRARY_ID`: Your Zotero library ID (for web API)
- `ZOTERO_LIBRARY_TYPE`: The type of library (user or group, default: user)
- `ZOTERO_LOCAL_API_KEY`: Local API key for writes (Zotero 10+). Normally set for you by
  `zotero-mcp authorize-local`; use the variable for containers and CI, where the config
  file isn't convenient
- `ZOTERO_LOCAL_SERVER_ID`: Pin the local Zotero database the key belongs to (optional —
  discovered automatically)
- `ZOTERO_LOCAL_WRITE`: `auto` (default) or `false` to force writes onto the web API even
  when a local key exists
- `ZOTERO_WEBDAV_URL`: Optional WebDAV folder URL for direct attachment downloads in remote mode
- `ZOTERO_WEBDAV_USERNAME`: Optional WebDAV username
- `ZOTERO_WEBDAV_PASSWORD`: Optional WebDAV password

Environment variables set in the shell you launch a client from (for example `claude`) override the values in its config file.

**Semantic search** (see [Semantic search](semantic-search.md)):
- `ZOTERO_EMBEDDING_MODEL`: Embedding model to use (default, openai, gemini, ollama)
- `OPENAI_API_KEY`: Your OpenAI API key (for OpenAI embeddings)
- `OPENAI_EMBEDDING_MODEL`: OpenAI model name (text-embedding-3-small, text-embedding-3-large)
- `OPENAI_BASE_URL`: Custom OpenAI endpoint URL (optional, for use with compatible APIs)
- OpenAI Batch API indexing is configured by `zotero-mcp setup` and can be overridden with
  `zotero-mcp update-db --openai-batch` or `--no-openai-batch`
- `GEMINI_API_KEY`: Your Gemini API key (for Gemini embeddings)
- `GEMINI_EMBEDDING_MODEL`: Gemini model name (gemini-embedding-001)
- `GEMINI_BASE_URL`: Custom Gemini endpoint URL (optional, for use with compatible APIs)
- `OLLAMA_EMBEDDING_MODEL`: Ollama embedding model name (qwen3-embedding by default)
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `ZOTERO_DB_PATH`: Custom `zotero.sqlite` path (optional). When unset, the
  database is located automatically: a data directory configured in Zotero's
  preferences (read from the profile's `prefs.js`) is tried first, then the
  default `~/Zotero` location.

**Read backend:**

In local mode (`ZOTERO_LOCAL=true`) read tools answer straight from `zotero.sqlite`
instead of paging the Zotero API. Measured on a 44,105-item library:

| operation | SQLite | API |
|---|---|---|
| one item | 0.4 ms | 43.5 ms |
| children of 25 items | 1.8 ms | 2,476 ms |
| all tags | 110 ms | 103.9 s |

Anything SQLite cannot express (a wildcard tag filter, a boolean `itemType`
expression) is answered through the API for that call, so the results are never
narrower than the API's. Writes always go through Zotero.

- `ZOTERO_BACKEND=api`: read through the Zotero API even in local mode, as before
  0.12.1. `ZOTERO_BACKEND=sqlite` forces SQLite; `ZOTERO_SEARCH_BACKEND` is
  accepted as an older name for the same setting. Outside local mode the API is
  always used.
- `ZOTERO_MCP_DB_SNAPSHOT_MIN_INTERVAL`: Zotero keeps recent changes in a WAL file
  next to `zotero.sqlite`, so reads use a private copy of the database plus that
  file. The copy is refreshed when Zotero writes, at most once per this many
  seconds (default `5`). `ZOTERO_MCP_DB_SNAPSHOT=0` reads the database in place
  instead, which never copies but misses changes until Zotero checkpoints.

**Global search across libraries:**

With the SQLite backend (the default in local mode), `zotero_search_items`, `zotero_advanced_search`
and `zotero_semantic_search` accept `search_all_libraries=True` (`--all-libraries`
on the CLI). One query then covers your personal library and every group library
at once, and each result is labelled with the library it came from:

```
**Library:** AI in entrepreneurship (groupID=6015547)
```

This is deliberately limited to the SQLite backend. The Zotero API can
only search one library per request, so without direct SQL the best anyone could
do is replay a single-library search against each library in turn — a different
and far slower operation. Rather than emulate global search badly, the tools
refuse and say so.

Two limits follow from how Zotero stores things. **Collections are per-library**
(`collections.libraryID` is NOT NULL), so `collection_key` and `collection`
conditions cannot be combined with a global search. **Tags are not** — Zotero
keeps one database-wide `tags` table shared by every library — so tag filters and
`tag` conditions work globally and are the recommended way to slice a global
search.

Duplicates across libraries are returned as-is: the same paper filed in two
libraries is two items, and collapsing them would hide where each copy lives.

**Tool surface:**
- `ZOTERO_MCP_TOOLSETS`: Which optional tool groups to expose. Every tool the
  server registers is sent to the model on *every* request, so the tool list is
  a fixed cost on your context window. Groups that need an external service,
  serve maintenance rather than research, or apply only to some users are off
  by default. See [Tool groups](tools.md#tool-groups).

**Item schema:**
- `ZOTERO_MCP_SCHEMA_REFRESH=0`: Disable the weekly background refresh of
  Zotero's item-type schema from `api.zotero.org`. The schema is what routes a
  generic `title=` update to the field a type actually stores it under (a
  statute's `nameOfAct`, a case's `caseName`). A copy ships with the package, so
  disabling the refresh only means new item types added by Zotero after this
  release won't be picked up until you upgrade. `zotero-mcp schema-refresh`
  still refreshes on demand.
- `ZOTERO_MCP_SCHEMA_CACHE`: Custom path for the refreshed schema cache
  (default: `~/.cache/zotero-mcp/schema.json`).

## Text extraction settings

PDFs are parsed with [pdf-inspector](https://github.com/firecrawl/pdf-inspector), which produces Markdown with the document's heading structure intact. These keys live under `semantic_search.extraction` in `~/.config/zotero-mcp/config.json`:

```json
{
  "semantic_search": {
    "extraction": {
      "pdf_max_pages": 50,
      "fulltext_display_max_pages": 10,
      "attachment_priority": ["markdown", "pdf", "html", "other"]
    }
  }
}
```

| Key | Default | What it does |
|---|---|---|
| `pdf_max_pages` | `50` | Pages extracted per PDF when indexing. Raising it does not widen what search sees on its own — that is bounded by the embedding model's token limit or `chunking.max_chunks_per_item`. |
| `fulltext_display_max_pages` | `10` | Pages returned by `zotero_get_item_fulltext`. Separate from the above because reading a paper is bounded by your assistant's context, not by recall. |
| `attachment_priority` | `["pdf", "html", "other"]` | Order in which attachment kinds are tried when an item has several readable files. |

**`attachment_priority`** exists for the case where you have converted a paper to clean Markdown yourself and attached it next to the original PDF. By default the PDF still wins; listing `"markdown"` first makes your converted copy the one that gets read and indexed. Valid entries are `pdf`, `html`, `markdown`, `text` and `other`. `other` is a catch-all matching every kind not named elsewhere in the list, so the default sweeps Markdown and plain text into one bucket where the larger file wins. Omitting `other` means anything unlisted is never chosen.

Changing this setting marks affected items for re-extraction, so a following `zotero-mcp update-db` refreshes text that came from a now-deprioritized attachment rather than leaving stale embeddings behind.

To read one specific attachment regardless of priority, pass that attachment's own key to `zotero_get_item_fulltext` (find it with `zotero_get_item_children`) — an attachment key bypasses the priority order and reads exactly that file.

Extracted text is reliable for prose and unreliable for math and tables. `zotero_read_pdf_pages` flags the pages where that happens and can return those pages as images (`format='image'`).

## Command-line options

```bash
# Run the server directly
zotero-mcp serve

# Specify transport method
zotero-mcp serve --transport stdio|streamable-http|sse

# Setup and configuration
zotero-mcp setup --help                    # Get help on setup options
zotero-mcp setup --semantic-config-only    # Configure only semantic search
zotero-mcp setup-info                      # Show installation path and config info for MCP clients
zotero-mcp authorize-local                 # Grant local writes (Zotero 10+)
zotero-mcp install-skill                   # Install the zotero-cli agent skill

# Updates and maintenance
zotero-mcp update                          # Update to latest version
zotero-mcp update --check-only             # Check for updates without installing
zotero-mcp update --force                  # Force update even if up to date

# Semantic search database management
zotero-mcp update-db                       # Update semantic search database (fast, metadata-only)
zotero-mcp update-db --openai-batch        # Submit OpenAI embeddings through Batch API
zotero-mcp update-db --no-openai-batch     # Force realtime OpenAI embeddings for this run
zotero-mcp openai-batch-status             # Check latest OpenAI embedding batch status
zotero-mcp openai-batch-import             # Import completed OpenAI batch embeddings
zotero-mcp update-db --fulltext            # Update with full-text extraction (comprehensive but slower)
zotero-mcp update-db --force-rebuild       # Force complete database rebuild
zotero-mcp update-db --fulltext --force-rebuild  # Rebuild with full-text extraction
zotero-mcp update-db --fulltext --db-path "your_path_to/zotero.sqlite" # Customize your zotero database path
zotero-mcp db-status                       # Show database status and info

# General
zotero-mcp version                         # Show current version
```
