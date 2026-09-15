# Semantic search

Find research by concept and meaning, not just keywords. Requires the `[semantic]` extra:

```bash
pip install "zotero-mcp-server[semantic]"
```

- **Vector-based similarity search** over your entire research library
- **Multiple embedding models**: Default (free, local), OpenAI, Gemini, and Ollama
- **Similarity scores** with each result
- **Auto-updating database** with configurable sync schedules

## Setup

During setup or separately, configure semantic search:

```bash
# Configure during initial setup (recommended)
zotero-mcp setup

# Or configure semantic search separately
zotero-mcp setup --semantic-config-only
```

**Available embedding models:**
- **Default (all-MiniLM-L6-v2)**: Free, runs locally, good for most use cases
- **OpenAI**: Better quality, requires API key (`text-embedding-3-small` or `text-embedding-3-large`)
- **Gemini**: Better quality, requires API key (`gemini-embedding-001`)
- **Ollama**: Runs locally via Ollama API (requires model name, e.g., 'qwen3-embedding')

### Ollama

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

Two `semantic_search.embedding_config` keys tune the Ollama path for slower
hardware or very large libraries:

```jsonc
"embedding_config": {
  "model_name": "bge-m3",
  "timeout": 600,            // HTTP timeout per /api/embed call (default 120s)
  "request_batch_size": 64   // documents per request (default 64)
}
```

Raise `timeout` if indexing reports `Read timed out`; lower
`request_batch_size` to make each request cover less GPU work, which usually
fixes timeouts more reliably than raising the timeout alone.

### OpenAI Batch API

When you choose OpenAI, setup also asks whether database updates should use
OpenAI Batch API. Batch updates are cheaper for large libraries, but they are
asynchronous: submit the batch, wait for completion, then import the embeddings.

### Update frequency

- **Manual**: Update only when you run `zotero-mcp update-db`
- **Auto on startup**: Update database every time the server starts
- **Daily**: Update once per day automatically
- **Every N days**: Set custom interval

## Building and updating the index

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

How much of each PDF is extracted, and which attachment is read when an item has several, is set in [Text extraction settings](configuration.md#text-extraction-settings).

## Example queries

In your AI assistant:
- *"Find research similar to machine learning concepts in neuroscience"*
- *"Papers that discuss climate change impacts on agriculture"*
- *"Research related to quantum computing applications"*
- *"Studies about social media influence on mental health"*
- *"Find papers conceptually similar to this abstract: [paste abstract]"*

From the shell: `zotero-cli search --mode semantic "attention mechanisms"`.

Problems building or querying the index: see [Troubleshooting](troubleshooting.md#semantic-search).
