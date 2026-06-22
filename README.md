# Zotero MCP: Chat with your Research Library—Local or Web—in Claude, ChatGPT, and more.

<p align="center">
  <a href="https://www.zotero.org/">
    <img src="https://img.shields.io/badge/Zotero-CC2936?style=for-the-badge&logo=zotero&logoColor=white" alt="Zotero">
  </a>
  <a href="https://www.anthropic.com/claude">
    <img src="https://img.shields.io/badge/Claude-6849C3?style=for-the-badge&logo=anthropic&logoColor=white" alt="Claude">
  </a>
  <a href="https://chatgpt.com/">
    <img src="https://img.shields.io/badge/ChatGPT-74AA9C?style=for-the-badge&logo=openai&logoColor=white" alt="ChatGPT">
  </a>
  <a href="https://modelcontextprotocol.io/introduction">
    <img src="https://img.shields.io/badge/MCP-0175C2?style=for-the-badge&logoColor=white" alt="MCP">
  </a>
  <a href="https://pypi.org/project/zotero-mcp-server/">
    <img src="https://img.shields.io/pypi/v/zotero-mcp-server?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://discord.gg/BvgjbcBUqg">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</p>

**Zotero MCP** seamlessly connects your [Zotero](https://www.zotero.org/) research library with [ChatGPT](https://openai.com), [Claude](https://www.anthropic.com/claude), and other AI assistants (e.g., [Cherry Studio](https://cherry-ai.com/), [Chorus](https://chorus.sh), [Cursor](https://www.cursor.com/)) via the [Model Context Protocol](https://modelcontextprotocol.io/introduction). Review papers, get summaries, analyze citations, extract PDF annotations, and more!

---

## ✨ Features

### 🧠 AI-Powered Semantic Search
- **Vector-based similarity search** over your entire research library (requires `[semantic]` extra)
- **Multiple embedding models**: Default (free, local), OpenAI, Gemini, and Ollama
- **Intelligent results** with similarity scores and contextual matching
- **Auto-updating database** with configurable sync schedules

### 🔍 Search Your Library
- Find papers, articles, and books by title, author, or content
- Perform complex searches with multiple criteria
- Browse collections, tags, and recent additions
- Semantic search for conceptual and topic-based discovery

### 📚 Access Your Content
- Retrieve detailed metadata for any item (markdown or BibTeX export)
- Get full text content (when available)
- Look up items by BetterBibTeX citation key

### 📝 Work with Annotations
- Extract and search PDF annotations with page numbers
- Access Zotero's native annotations
- Create and update notes and annotations
- Extract PDF table of contents / outlines (requires `[pdf]` extra)

### ✏️ Write Operations
- **Add papers by DOI** with auto-fetched metadata and open-access PDF cascade (Unpaywall, arXiv, Semantic Scholar, PMC)
- **Add papers by URL** (arXiv, DOI links, generic webpages) or from local files
- Create and manage collections, update item metadata, batch-update tags
- Find and merge duplicate items with dry-run preview
- **Hybrid mode**: local reads + web API writes for local-mode users

### 📊 Scite Citation Intelligence (optional `[scite]` extra)
- **Citation tallies**: See how many papers support, contrast, or mention each item — the MCP version of the [Scite Zotero Plugin](https://github.com/scitedotai/scite-zotero-plugin)
- **Retraction alerts**: Scan your library for retracted or corrected papers
- No Scite account required — uses public API endpoints

### 🌐 Flexible Access Methods
- Local mode for offline access (no API key needed)
- Web API for cloud library access
- Hybrid mode: read from local Zotero, write via web API

### ⌨️ Standalone CLI (`zotero-cli`)
- Search, browse, and edit your library directly from the terminal — no AI assistant required
- Ideal for scripting, automation, and quick lookups
- Short aliases (`s`, `g`, `ann`, `coll`) for interactive use

## 🚀 Quick Install

> **New to the command line?** Try the community-built [Zotero MCP Setup](https://github.com/ehawkin/zotero-mcp-setup) — includes a macOS GUI installer (DMG), one-click install scripts for Mac/Windows, and a step-by-step guide. No Terminal experience needed.

### Default Installation (core tools only)

The base install is lightweight — it includes search, metadata retrieval, annotations, and write operations. No ML/AI dependencies are pulled in.

#### Installing via uv (recommended)

```bash
uv tool install zotero-mcp-server
zotero-mcp setup  # Auto-configure (Claude Desktop supported)
```

#### Installing via pip

```bash
pip install zotero-mcp-server
zotero-mcp setup  # Auto-configure (Claude Desktop supported)
```

#### Installing via pipx

```bash
pipx install zotero-mcp-server
zotero-mcp setup  # Auto-configure (Claude Desktop supported)
```

### Optional Extras

Heavy ML/PDF dependencies are separated into optional extras so the base install stays fast and small:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `semantic` | Semantic search via ChromaDB, sentence-transformers, OpenAI/Gemini embeddings | `pip install "zotero-mcp-server[semantic]"` |
| `pdf` | PDF outline extraction (PyMuPDF) and EPUB annotation support | `pip install "zotero-mcp-server[pdf]"` |
| `scite` | [Scite](https://scite.ai) citation intelligence — tallies and retraction alerts (no account needed) | `pip install "zotero-mcp-server[scite]"` |
| `all` | Everything above | `pip install "zotero-mcp-server[all]"` |

For example, with uv:
```bash
uv tool install "zotero-mcp-server[all]"    # Full install with all features
uv tool install "zotero-mcp-server[semantic]" # Just semantic search
```

If you only need basic library access (search, read, annotate, write), the default install with no extras is all you need.

#### Updating Your Installation

Keep zotero-mcp up to date with the smart update command:

```bash
# Check for updates
zotero-mcp update --check-only

# Update to latest version (preserves all configurations)
zotero-mcp update
```

## 🧠 Semantic Search

Zotero MCP now includes powerful AI-powered semantic search capabilities that let you find research based on concepts and meaning, not just keywords.

### Setup Semantic Search

During setup or separately, configure semantic search:

```bash
# Configure during initial setup (recommended)
zotero-mcp setup

# Or configure semantic search separately
zotero-mcp setup --semantic-config-only
```

**Available Embedding Models:**
- **Default (all-MiniLM-L6-v2)**: Free, runs locally, good for most use cases
- **OpenAI**: Better quality, requires API key (`text-embedding-3-small` or `text-embedding-3-large`)
- **Gemini**: Better quality, requires API key (`gemini-embedding-001`)
- **Ollama**: Runs locally via Ollama API (requires model name, e.g., 'qwen3-embedding')

**Using Ollama embeddings:**

Install and start Ollama, then pull an embedding model before running `zotero-mcp update-db`:

```bash
ollama serve

# Small model: fast and lightweight
ollama pull nomic-embed-text

# Medium model: better multilingual retrieval quality
ollama pull bge-m3
```

When prompted by `zotero-mcp setup --semantic-config-only`, choose **Ollama** and use either `nomic-embed-text` or `bge-m3` as the model name. If you change embedding models later, rebuild the index:

```bash
zotero-mcp update-db --force-rebuild
```

When you choose OpenAI, setup also asks whether database updates should use
OpenAI Batch API. Batch updates are cheaper for large libraries, but they are
asynchronous: submit the batch, wait for completion, then import the embeddings.

**Update Frequency Options:**
- **Manual**: Update only when you run `zotero-mcp update-db`
- **Auto on startup**: Update database every time the server starts
- **Daily**: Update once per day automatically
- **Every N days**: Set custom interval

### Using Semantic Search

After setup, initialize your search database:

```bash
# Build the semantic search database (fast, metadata-only)
zotero-mcp update-db

# Submit OpenAI embeddings through Batch API for this update
zotero-mcp update-db --openai-batch

# Check and import completed OpenAI Batch API embeddings
zotero-mcp openai-batch-status
zotero-mcp openai-batch-import

# Force realtime OpenAI embeddings even if Batch API is enabled in config
zotero-mcp update-db --no-openai-batch

# Build with full-text extraction (slower, more comprehensive)
zotero-mcp update-db --fulltext

# Use your custom zotero.sqlite path
zotero-mcp update-db --fulltext --db-path "/Your_custom_path/zotero.sqlite"

# If you have embedding conflicts or changed models, force a rebuild
zotero-mcp update-db --force-rebuild

# Check database status
zotero-mcp db-status
```

**Example Semantic Queries in your AI assistant:**
- *"Find research similar to machine learning concepts in neuroscience"*
- *"Papers that discuss climate change impacts on agriculture"*
- *"Research related to quantum computing applications"*
- *"Studies about social media influence on mental health"*
- *"Find papers conceptually similar to this abstract: [paste abstract]"*

The semantic search provides similarity scores and finds papers based on conceptual understanding, not just keyword matching.

## 🖥️ Setup & Usage

Full documentation is available at [Zotero MCP docs](https://stevenyuyy.com/zotero-mcp/).

**Requirements**
- Python 3.10+
- Zotero 7+ (for local API with full-text access)
- An MCP-compatible client (e.g., Claude Desktop, ChatGPT Developer Mode, Cherry Studio, Chorus)

**For ChatGPT setup: see the [Getting Started guide](./docs/getting-started.md).**

### For Claude Desktop (example MCP client)

#### Configuration
After installation, either:

1. **Auto-configure** (recommended):
   ```bash
   zotero-mcp setup
   ```

2. **Manual configuration**:
   Add to your `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "zotero": {
         "command": "zotero-mcp",
         "env": {
           "ZOTERO_LOCAL": "true",
           "ZOTERO_API_KEY": "YOUR_API_KEY",
           "ZOTERO_LIBRARY_ID": "YOUR_LIBRARY_ID"
         }
       }
     }
   }
   ```

   For **local read-only use**, `ZOTERO_LOCAL: "true"` is all you need — drop the
   `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID` lines entirely. Add them only to enable
   **write mode**: the local API is fast but read-only, so the server uses the Zotero
   web API for write operations.

   - Generate an API key from <https://www.zotero.org/settings/security#applications>.
   - `ZOTERO_LIBRARY_ID` is your numeric **userID**, shown on that same page (for a
     group library, use the group's ID and also set `ZOTERO_LIBRARY_TYPE: "group"`).

   > **Tip:** if Claude Desktop reports it can't find the `zotero-mcp` command, use the
   > absolute path instead (run `zotero-mcp setup-info` or `which zotero-mcp` to find it) —
   > GUI apps don't always inherit your shell `PATH`.

#### Usage

1. Start Zotero desktop (make sure local API is enabled in preferences)
2. Launch Claude Desktop
3. Access the Zotero-MCP tool through Claude Desktop's tools interface

Example prompts:
- "Search my library for papers on machine learning"
- "Find recent articles I've added about climate change"
- "Summarize the key findings from my paper on quantum computing"
- "Extract all PDF annotations from my paper on neural networks"
- "Search my notes and annotations for mentions of 'reinforcement learning'"
- "Show me papers tagged '#Arm' excluding those with '#Crypt' in my library"
- "Search for papers on operating system with tag '#Arm'"
- "Export the BibTeX citation for papers on machine learning"
- **"Find papers conceptually similar to deep learning in computer vision"** *(semantic search)*
- **"Research that relates to the intersection of AI and healthcare"** *(semantic search)*
- **"Papers that discuss topics similar to this abstract: [paste text]"** *(semantic search)*

### For Cherry Studio

#### Configuration
Go to Settings -> MCP Servers -> Edit MCP Configuration, and add the following:

```json
{
  "mcpServers": {
    "zotero": {
      "name": "zotero",
      "type": "stdio",
      "isActive": true,
      "command": "zotero-mcp",
      "args": [],
      "env": {
        "ZOTERO_LOCAL": "true"
      }
    }
  }
}
```
Then click "Save".

Cherry Studio also provides a visual configuration method for general settings and tools selection.

## 🔧 Advanced Configuration

### Using Web API Instead of Local API

For accessing your Zotero library via the web API (useful for remote setups):

```bash
zotero-mcp setup --no-local --api-key YOUR_API_KEY --library-id YOUR_LIBRARY_ID
```

### Environment Variables

**Zotero Connection:**
- `ZOTERO_LOCAL=true`: Use the local Zotero API (default: false)
- `ZOTERO_API_KEY`: Your Zotero API key (for web API)
- `ZOTERO_LIBRARY_ID`: Your Zotero library ID (for web API)
- `ZOTERO_LIBRARY_TYPE`: The type of library (user or group, default: user)
- `ZOTERO_WEBDAV_URL`: Optional WebDAV folder URL for direct attachment downloads in remote mode
- `ZOTERO_WEBDAV_USERNAME`: Optional WebDAV username
- `ZOTERO_WEBDAV_PASSWORD`: Optional WebDAV password

**Semantic Search:**
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
- `ZOTERO_DB_PATH`: Custom `zotero.sqlite` path (optional)

### Command-Line Options

```bash
# Run the server directly
zotero-mcp serve

# Specify transport method
zotero-mcp serve --transport stdio|streamable-http|sse

# Setup and configuration
zotero-mcp setup --help                    # Get help on setup options
zotero-mcp setup --semantic-config-only    # Configure only semantic search
zotero-mcp setup-info                      # Show installation path and config info for MCP clients

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
zotero-mcp update-db --fulltext             # Update with full-text extraction (comprehensive but slower)
zotero-mcp update-db --force-rebuild       # Force complete database rebuild
zotero-mcp update-db --fulltext --force-rebuild  # Rebuild with full-text extraction
zotero-mcp update-db --fulltext --db-path "your_path_to/zotero.sqlite" # Customize your zotero database path
zotero-mcp db-status                       # Show database status and info

# General
zotero-mcp version                         # Show current version
```

## ⌨️ CLI Mode (`zotero-cli`)

`zotero-cli` is a standalone terminal interface to your Zotero library. It uses the same tools as the MCP server but without needing an AI assistant — useful for quick lookups, shell scripts, and automation.

Use `zotero-mcp` when your AI client supports MCP (Claude Desktop, ChatGPT). Use `zotero-cli` for shell scripts, cron jobs, or agentic pipelines with shell access (e.g. Claude Code) — CLI commands cost far fewer tokens than MCP tool schemas and compose naturally with Unix pipes.

Both share the same configuration set up by `zotero-mcp setup`.

### Quick reference

```bash
# Search
zotero-cli search "machine learning"           # keyword search
zotero-cli s "neural networks" --limit 5       # short alias, limit results
zotero-cli search --mode semantic "attention mechanisms"
zotero-cli search --mode tag "important,reviewed"

# Get item details
zotero-cli get metadata ABC123                 # markdown metadata
zotero-cli g metadata ABC123 --format bibtex  # BibTeX export
zotero-cli get fulltext ABC123                 # full text
zotero-cli get children ABC123                 # attachments and notes

# Edit item metadata
zotero-cli edit ABC123 --title "New Title"
zotero-cli edit ABC123 --add-tags "reviewed,important" --date "2024"

# Notes and annotations
zotero-cli notes list ABC123
zotero-cli notes create --item-key ABC123 --text "My note" --tags "idea"
zotero-cli notes create --item-key ABC123 --text -   # read from stdin
zotero-cli ann list ABC123                    # annotations (short alias)
zotero-cli ann search "highlight text"

# Add items
zotero-cli add doi 10.1038/s41586-021-03819-2
zotero-cli add url https://arxiv.org/abs/2301.00001
zotero-cli add file --filepath /path/to/paper.pdf --title "Override Title"
zotero-cli add isbn 9780262046305
zotero-cli add bibtex --file refs.bib                # or --bibtex '@article{...}'
zotero-cli add bibtex --bibtex - < refs.bib          # stdin via -
zotero-cli add csl-json --file refs.json             # or --json '...' / --json -

# --collections accepts keys, names, or parent/child paths — resolved and
# validated before the item is created (a typo fails the add, with suggestions,
# instead of leaving an unfiled item)
zotero-cli add doi 10.1038/s41586-021-03819-2 --collections "Reading List"
zotero-cli collections manage --item-keys ABC123 --add-to "_project/topic"

# Adds are idempotent by default (--if-exists file): if the item is already in
# the library it is reused — filed into any missing collections, given any
# missing tags — instead of duplicated. Re-running the same command is a no-op.
zotero-cli add doi 10.1038/s41586-021-03819-2 -c "Reading List"   # run it twice: converges
zotero-cli add doi 10.1038/s41586-021-03819-2 --if-exists skip       # never touch existing
zotero-cli add doi 10.1038/s41586-021-03819-2 --if-exists duplicate  # old behavior
zotero-cli add doi 10.1038/s41586-021-03819-2 -c "New Topic" --create-collections
# -c/--collection is repeatable and never comma-split (names with commas work);
# --collections remains the comma-separated form

# Collections and tags
zotero-cli coll list                          # list collections (short alias)
zotero-cli coll search "PhD Research"
zotero-cli tags list

# Semantic search database
zotero-cli db update
zotero-cli db update --fulltext --force-rebuild
zotero-cli db status

# Library and duplicates
zotero-cli library info
zotero-cli duplicates find
```

### Verbose mode

Add `-v` anywhere to see progress messages (e.g., which API calls are made):

```bash
zotero-cli -v search "CRISPR"
```

## 📑 PDF Annotation Extraction

Zotero MCP includes advanced PDF annotation extraction capabilities:

- **Direct PDF Processing**: Extract annotations directly from PDF files, even if they're not yet indexed by Zotero
- **Enhanced Search**: Search through PDF annotations and comments
- **Image Annotation Support**: Extract image annotations from PDFs
- **Seamless Integration**: Works alongside Zotero's native annotation system

For optimal annotation extraction, it is **highly recommended** to install the [Better BibTeX plugin](https://retorque.re/zotero-better-bibtex/installation/) for Zotero. The annotation-related functions have been primarily tested with this plugin and provide enhanced functionality when it's available.


The first time you use PDF annotation features, the necessary tools will be automatically downloaded.

## 🔗 Managing Related Items

Zotero MCP now supports managing relationships between items in your library. This is useful for linking related papers, tracking versions, or connecting preprints to their published versions.

### View Related Items
```
zotero_get_item_related(item_key="ABCD1234")
```

### Add a Relation
Create a bidirectional link between two items:
```
zotero_add_item_relation(
    item_key="ABCD1234",
    related_item_key="EFGH5678",
    relation_type="dc:relation"  # Optional, defaults to "dc:relation"
)
```

### Remove a Relation
```
zotero_remove_item_relation(
    item_key="ABCD1234",
    related_item_key="EFGH5678",
    remove_bidirectional=True  # Also remove the reverse relation (default: true)
)
```

**Relation Types:**
- `dc:relation` — General related items (default)
- `owl:sameAs` — Items that are the same work (e.g., preprint and published version)

## 📚 Available Tools

### 🧠 Semantic Search Tools
- `zotero_semantic_search`: AI-powered similarity search with embedding models
- `zotero_update_search_database`: Manually update the semantic search database
- `zotero_get_search_database_status`: Check database status and configuration

### 🔍 Search Tools
- `zotero_search_items`: Search your library by keywords
- `zotero_advanced_search`: Perform complex searches with multiple criteria
- `zotero_get_collections`: List collections
- `zotero_get_collection_items`: Get items in a collection
- `zotero_get_tags`: List all tags
- `zotero_get_recent`: Get recently added items
- `zotero_search_by_tag`: Search your library using custom tag filters

### 📚 Content Tools
- `zotero_get_item_metadata`: Get detailed metadata (supports `format="markdown"`, `format="json"` for complete raw Zotero metadata, and `format="bibtex"`)
- `zotero_get_item_fulltext`: Get full text content
- `zotero_get_item_children`: Get attachments and notes

### 📝 Annotation & Notes Tools
- `zotero_get_annotations`: Get annotations (including direct PDF extraction)
- `zotero_get_notes`: Retrieve notes from your Zotero library
- `zotero_search_notes`: Search in notes and annotations (including PDF-extracted)
- `zotero_create_note`: Create a new note for an item (beta feature)
- `zotero_get_page_layout`: Detect figure/table regions on a PDF page (with captions and normalized coordinates) for accurate area annotation placement

### 📊 Scite Citation Intelligence Tools
- `scite_enrich_item`: Get Scite citation tallies and retraction alerts for a paper
- `scite_enrich_search`: Search your Zotero library with Scite-enriched results (tallies + alerts inline)
- `scite_check_retractions`: Scan items for retractions and editorial notices

### 📦 Item & Collection Management Tools
- `zotero_add_by_doi`: Add a paper by DOI with automatic metadata and open-access PDF attachment
- `zotero_add_by_url`: Add a paper by URL (arXiv, DOI URLs, and general webpages)
- `zotero_add_by_isbn`: Add a book by ISBN (Open Library + Google Books cascade)
- `zotero_add_by_bibtex`: Add one or more items from BibTeX (inline or .bib file)
- `zotero_add_by_csl_json`: Add one or more items from CSL JSON (inline or file)
- `zotero_add_from_file`: Import a local PDF or EPUB file with automatic DOI extraction

All add tools take a `collections` parameter accepting collection keys, names, or `parent/child` paths — resolved and validated before the item is created, so unknown or ambiguous specs fail with suggestions instead of producing an unfiled item. They also take `if_exists` (`"duplicate"` — default — always creates; `"file"` reuses an existing item matching the DOI/arXiv ID/ISBN/URL, filing it into missing collections and adding missing tags; `"skip"` leaves a match untouched) and `create_missing_collections` (create unknown collection specs, including path chains, instead of failing). The `zotero-cli add` commands default to `--if-exists file`.
- `zotero_create_collection`: Create a new collection (folder/project) in your library
- `zotero_search_collections`: Search for collections by name to find their keys
- `zotero_manage_collections`: Add or remove items from collections (accepts keys, names, or `parent/child` paths)
- `zotero_update_item`: Update metadata for an existing item (title, tags, abstract, date, etc.)
- `zotero_find_duplicates`: Find duplicate items by title and/or DOI
- `zotero_merge_duplicates`: Merge duplicate items with dry-run preview; consolidates all child items
- `zotero_get_pdf_outline`: Extract the table of contents / outline from a PDF attachment
- `zotero_search_by_citation_key`: Look up items by BetterBibTeX citation key (with Extra field fallback)

### 🔗 Related Items Tools
- `zotero_get_item_related`: Get all related items for a specific Zotero item
- `zotero_add_item_relation`: Add a related item relationship (creates bidirectional link)
- `zotero_remove_item_relation`: Remove a related item relationship

## 🧪 Testing

### Unit Tests
```bash
uv run pytest tests/     # 294 tests, ~2 seconds
```

### Integration Test Plan
A 45-point live integration test plan is included at `docs/integration-test-plan.md`. It's designed to be given to Claude in Claude Desktop, which will execute each test against your real Zotero library. Tests cover all tools, PDF attachment cascade, attach_mode, BetterBibTeX lookups, and multi-step showcase prompts. See the file for full instructions.

## 🔍 Troubleshooting

### General Issues
- **No results found**: Ensure Zotero is running and the local API is enabled. You need to toggle on `Allow other applications on this computer to communicate with Zotero` in Zotero preferences.
- **Can't connect to library**: Check your API key and library ID if using web API
- **Full text not available**: Make sure you're using Zotero 7+ for local full-text access
- **Local library limitations**: Some functionality (tagging, library modifications) may not work with local JS API. Consider using web library setup for full functionality. (See the [docs](docs/getting-started.md#local-library-limitations) for more info.)
- **Installation/search option switching issues**: Database problems from changing install methods or search options can often be resolved with `zotero-mcp update-db --force-rebuild`

### Semantic Search Issues
- **"Missing required environment variables" when running update-db**: Run `zotero-mcp setup` to configure your environment, or the CLI will automatically load settings from your MCP client config (e.g., Claude Desktop)
- **ChromaDB / stale embedding model errors**: If you changed embedding models and see 404 errors (e.g., `text-embedding-004 is not found`), run `zotero-mcp update-db --force-rebuild` to recreate the collection with your current model. If that doesn't work, delete `~/.config/zotero-mcp/chroma_db/` and rebuild.
- **Database update takes long**: By default, `update-db` is fast (metadata-only). For comprehensive indexing with full-text, use `--fulltext` flag. Use `--limit` parameter for testing: `zotero-mcp update-db --limit 100`
- **Semantic search returns no results**: Ensure the database is initialized with `zotero-mcp update-db` and check status with `zotero-mcp db-status`
- **Limited search quality**: For better semantic search results, use `zotero-mcp update-db --fulltext` to index full-text content (requires local Zotero setup)
- **OpenAI/Gemini API errors**: Verify your API keys are correctly set and have sufficient credits/quota

### Update Issues
- **Update command fails**: Check your internet connection and try `zotero-mcp update --force`
- **Configuration lost after update**: The update process preserves configs automatically, but check `~/.config/zotero-mcp/` for backup files

## ☕ Support

If you find Zotero MCP useful, consider buying me a coffee!

<a href="https://buymeacoffee.com/stevenyuyy">
  <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">
</a>

## 📄 License

MIT
