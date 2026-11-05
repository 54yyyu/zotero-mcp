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
See [How batch runs are collected](#how-batch-runs-are-collected).

### Update frequency

- **Manual**: Update only when you run `zotero-mcp update-db`
- **Auto on startup**: Update database every time the server starts
- **Daily**: Update once per day automatically
- **Every N days**: Set custom interval

Automatic updates run when the server starts and before a semantic search. If you use Batch API embeddings and rarely search, schedule `zotero-mcp update-db` with cron so that finished batches get imported.

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

### How batch runs are collected

Results of a batch run (OpenAI or Gemini) are already paid for, so every `update-db`, the `zotero_update_search_database` tool and every automatic update first looks for unfinished runs of the active library. It imports the files that have finished and submits the ones the enqueued-token budget held back, even if the update itself runs in realtime. With no unfinished run this only reads local files and makes no provider call. `batch-import` does the same for one run, and starts nothing new.

- **A new submission waits behind an unfinished run.** The update says which run it is waiting for, and the sync version stays where it is, so its changes go in with the next update after that run has finished. Submitting anyway would pay again for the items in the unfinished run. Only `update-db --force-rebuild` replaces an unfinished run, unless `--limit` is also given.
- **While a run is unfinished, the check before a semantic search only collects.** It runs at most hourly instead of on the daily schedule, and never starts a new run. With auto-update off it does nothing, so collect by hand with `update-db` or `batch-import`.
- **A failed chunk doesn't block anything.** The update reports it, and since the sync version didn't move, the next update submits those items again. A force rebuild imports all-or-nothing, so one failed chunk ends that run; the index stays as it was and the update reports what to run.
- **An import doesn't lower the sync version.** The exception is a force rebuild as it lands: it has just replaced the ChromaDB collection, so the sync version has to say what the collection now holds.
- **An unattended import of a force rebuild won't drop other libraries' documents.** If another library was indexed into the ChromaDB collection after you submitted the rebuild, the update tells you to run `zotero-mcp batch-import`, which imports it anyway. A rebuild you submitted with `--allow-mass-deletion` doesn't ask again.

A run that can't be finished at the provider, for example because its API key changed, keeps being reported. To give it up, delete its directory under `~/.config/zotero-mcp/openai_batches/` or `gemini_batches/`; the update's message names it. Results made with a different embedding model aren't imported automatically, since no query in the current ChromaDB collection could match them.

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

The label is ordinary chunk metadata, so it can be filtered on:

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
