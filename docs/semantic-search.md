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

### Instructions for query embeddings

For instruction-tuned models such as Qwen3-Embedding, set an optional
`query_instruction` under `semantic_search.embedding_config`. This works with
`embedding_model: "ollama"` and HuggingFace models (`"qwen"` or a HuggingFace
model name):

```json
{
  "semantic_search": {
    "embedding_model": "ollama",
    "embedding_config": {
      "model_name": "qwen3-embedding:0.6b",
      "query_instruction": "Given a search query, retrieve relevant academic papers, books and reports from a research library"
    }
  }
}
```

Queries become `Instruct: <instruction>\nQuery: <query>`. Document text is
unchanged, so changing only this setting does not require re-indexing. Omit it,
set it to `null`, or use an empty/whitespace-only string to keep plain queries.
The setting is saved with the embedding function's configuration and restored
when the collection is reopened. Instructions are per embedding-function
instance, even when HuggingFace instances share the same loaded model. No
instruction is selected automatically; choose one appropriate for your model.

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

For Ollama and OpenAI-compatible servers, `max_input_tokens` caps the text sent
per document (default 8000). Set it below the server's limit when indexing
reports an input that is too large, for example llama-server's
`input (N tokens) is too large to process`. It is counted with OpenAI's
tokenizer, so leave headroom for the model's own: `3000` for a model served
with a 4096-token batch.

### OpenAI Batch API

When you choose OpenAI, setup also asks whether database updates should use
OpenAI Batch API. Batch updates are cheaper for large libraries, but they are
asynchronous: submit the batch, wait for completion, then import the embeddings.

[Passage chunking](configuration.md#passage-chunking) works on this path too, and a large library needs a little planning. Providers limit how many tokens can be queued at once, so a run is split into files and the ones that don't fit wait for earlier ones to finish: `update-db --auto-loop` keeps that going, or run `batch-import` from time to time. After two days a run no longer holds back the next update, and if one starts, the older run's unsent files are dropped; what already landed is kept, and the next `update-db` adds the passages the dropped files would have. The default budgets are the lowest tier's, so set `batch_max_enqueued_tokens` under `semantic_search.<provider>_batch` a little below your account's limit. Also consider `batch_max_requests: 10000`, because each output file is read into memory during import and an output row is about 65 KB at 3072 dimensions.

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

# Rebuild an unreadable index from its own files, without re-embedding
zotero-mcp repair-db
```

`repair-db` is for an index that ChromaDB can no longer read, typically after an interrupted update; see [Repairing an unreadable index](troubleshooting.md#repairing-an-unreadable-index).

How much of each PDF is extracted, and which attachment is read when an item has several, is set in [Text extraction settings](configuration.md#text-extraction-settings).

### Index location

The index is stored in `~/.config/zotero-mcp/chroma_db`. To keep it elsewhere, set `persist_directory` in `~/.config/zotero-mcp/config.json`:

```json
{
  "semantic_search": {
    "persist_directory": "D:/zotero-mcp/chroma_db"
  }
}
```

On Windows the path must be ASCII-only. If your user folder name has other characters (for example `C:\Users\王林澜`), ChromaDB writes part of the index to the wrong place once it holds about 1,000 items, and every later search fails with `Error loading hnsw index`. Point `persist_directory` at an ASCII path and run `zotero-mcp update-db --force-rebuild`.

`zotero-mcp setup` keeps this setting. Moving it starts an empty index unless you move the old `chroma_db` folder there too.

## Section labels

With [passage chunking](configuration.md#passage-chunking) on, each passage
records the section of the paper it falls in (Abstract,
Introduction, Methods, Results, Discussion, Conclusion, References, Appendix or
Back matter), and search results show it first in the **Location** line. A hit
in someone's Introduction is a citation of a finding; a hit in their Results is
the finding.

The label is stored with the passage, so it can be filtered on (see [Filters](#filters) for the fields that are not):

```python
zotero_semantic_search(query="...", filters={"section": {"$ne": "Introduction"}})
zotero_semantic_search(query="...", filters={"section": {"$in": ["Results", "Discussion"]}})
```

Headings are recognised by shape (a line of their own, optional numbering, an
optional colon), because extracted PDF text keeps line breaks but not styling.
A heading the parser does not recognise ends the current label rather than
letting it run on, so a passage is left unlabelled rather than mislabelled.
Two consequences:

- **An existing index has no labels** until it is rebuilt with
  `zotero-mcp update-db --fulltext --force-rebuild`. Items indexed or
  re-indexed after upgrading get them as they go.
- **Books mostly come out unlabelled.** Chapter titles are not section names,
  and many book PDFs extract without any heading the parser can read.
  Unlabelled passages are still searched; only a `section` filter excludes
  them.

## Filters

`zotero_semantic_search` and `zotero-cli search --mode semantic --filters` take a ChromaDB-style `where` dict. Two kinds of field can appear in it.

Stored in the index and filtered by ChromaDB: `group_id`, `section`, `page`, `chunk_index`, `n_chunks`, `char_start`, `char_end`.

Read from Zotero when the search runs: `item_type`, `item_key`, `title`, `doi`, `citation_key`, `publication`, `tags` and `has_fulltext`. These are evaluated against the current data of the result's parent item (the item itself when it has none), not against anything copied into the index, so an edit in Zotero applies to the next search without re-indexing. `item_key` is the parent's key. `tags` is a membership test: `{"tags": "to-read"}` matches items that have that tag among others, `{"tags": {"$in": ["a", "b"]}}` matches items with either, and `$ne` and `$nin` match items without it. `has_fulltext` is true when a hit came from a file's text; it is judged from the hits of the search, so an item found only through its own record reads as false.

```python
zotero_semantic_search(query="...", filters={"item_type": "journalArticle", "tags": "to-read"})
```

Several fields in one dict must all match. A condition that mixes stored and Zotero fields under `$or` is evaluated against Zotero as a whole. Records written by earlier versions still carry copies of some of these fields; they are not consulted.

How a filter on these fields is applied depends on the mode. In local mode zotero.sqlite names the matching items and ChromaDB only returns their records. In web mode, a filter made of `item_type`, `tags` and `item_key` conditions (equal to, or in a list, joined by AND) is sent to the Zotero API in one keys-only request per library, and ChromaDB returns only records of the matching items or of attachments whose `parent_hint` names one; the hint is the attachment's parent when it was indexed, used only for this, so an attachment moved since then can be missed by a filtered search until its record is next rewritten, but a hit is always resolved and checked against Zotero before it is shown. Any other filter, an index with records from before the hint (`update-db` or `migrate-index` sets it without embedding anything), or a failed request searches first and filters the hits after, asking again for more candidates up to three times.

Results show the parent item. When the hit was in one of several files, or in a standalone attachment, the Location line starts with `Attachment:` and the file's name.

## Re-ranking

A re-ranker reads the query and each candidate passage together and scores how
well the passage answers the query — slower than comparing embeddings, but
more accurate, so it is applied to the shortlist the embedding search
returns. `candidate_multiplier` sets the shortlist: papers per wanted result.

A local cross-encoder runs on your machine (needs `sentence-transformers`; the
strong models want a GPU):

```jsonc
"reranker": {
  "enabled": true,
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "candidate_multiplier": 3
}
```

A hosted re-ranker answers in well under a second without a GPU. Set
`provider` to `voyage`, `cohere`, `openrouter` or `contextual`, and put the
provider's key in the environment the server runs in (`VOYAGE_API_KEY`,
`COHERE_API_KEY`, `OPENROUTER_API_KEY`, `CONTEXTUAL_API_KEY`):

```jsonc
"reranker": {
  "enabled": true,
  "provider": "voyage",
  "model": "rerank-3",
  "candidate_multiplier": 5,
  "instruction": "Prefer passages that report findings over ones that cite them."  // optional
}
```

Any other endpoint that takes `{model, query, documents, top_n}` and answers
with `{index, relevance_score}` records works with `base_url` and
`api_key_env` in place of `provider`. `timeout` (seconds, default 30) bounds
each call; one retry follows a 429 or 5xx answer.

**Privacy:** with a hosted provider, every search sends the query and the text
of the candidate passages — dozens of excerpts from your library — to that
provider. The local cross-encoder sends nothing anywhere.

Search results say how they were ordered: *"Ranked by voyage/rerank-3 over 80
candidates"*, with the re-ranker's score as **Relevance** and the
embedding similarity alongside. If a hosted call fails — a wrong model name,
an expired key, a timeout — the search still answers, in embedding order, and
says why: *"re-ranking with voyage/rerank-3 failed (HTTP 401 …)"*.

## Example queries

In your AI assistant:
- *"Find research similar to machine learning concepts in neuroscience"*
- *"Papers that discuss climate change impacts on agriculture"*
- *"Research related to quantum computing applications"*
- *"Studies about social media influence on mental health"*
- *"Find papers conceptually similar to this abstract: [paste abstract]"*

From the shell: `zotero-cli search --mode semantic "attention mechanisms"`.

Problems building or querying the index: see [Troubleshooting](troubleshooting.md#semantic-search).
