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

[Passage chunking](configuration.md#passage-chunking) works on this path too, and a large library needs a little planning. Providers limit how many tokens can be queued at once, so a run is split into files and the ones that don't fit wait for earlier ones to finish: `update-db --auto-loop` keeps that going, or run `batch-import` from time to time. The default budgets are the lowest tier's, so set `batch_max_enqueued_tokens` under `semantic_search.<provider>_batch` a little below your account's limit. Also consider `batch_max_requests: 10000`, because each output file is read into memory during import and an output row is about 65 KB at 3072 dimensions. A file or row that fails is retried up to three times. The local copy of each file is deleted once it is imported.

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

If the embedding API cannot be reached, for example because it only accepts requests from a campus network or VPN, `update-db` stops after one short test request instead of working through the library; see [Embedding API unreachable](troubleshooting.md#embedding-api-unreachable).

`repair-db` is for an index that ChromaDB can no longer read, typically after an interrupted update; see [Repairing an unreadable index](troubleshooting.md#repairing-an-unreadable-index).

### How batch runs are collected

Results of a batch run (OpenAI or Gemini) are already paid for, so every `update-db`, the `zotero_update_search_database` tool and every automatic update first looks for unfinished runs of the active library. It imports the files that have finished and submits the ones the enqueued-token budget held back, even if the update itself runs in realtime. With no unfinished run this only reads local files and makes no provider call. `batch-import` does the same for one run, and starts nothing new.

- **An update waits behind an unfinished run.** That goes for a new batch submission and for a realtime update, including the ones cron, the server's startup or `zotero_update_search_database` run, since either would pay again for the run's items. The update says which run it is waiting for, and the sync version stays where it is, so its changes go in after that run. `update-db --no-batch` goes ahead anyway and warns that the run's results will be discarded. `update-db --force-rebuild` without `--limit` replaces the run, and warns that it does. A run you gave up on, or one that can't finish, replaces nothing: an older run's results still go in.
- **A run stops holding things up when nothing more is coming from it.** That's when the provider answers that it no longer has a batch or file, or a day past the provider's window (24 hours for OpenAI, 48 for Gemini), counted from the last chunk sent. Updates go ahead and say so. A wrong API key or base URL gets the same answer, so it isn't final: `batch-import` asks the provider again, `batch-status` never counts that answer, and a chunk whose results are already downloaded is never given up. Past the window an update still imports finished files if it can, but it doesn't send the run's waiting chunks unless you run `batch-import`. `zotero-mcp batch-abandon` gives a run up at any time; `--run-id` chooses when there are several. Its items are covered again by the next update. Giving up a force rebuild that had already started replacing the index sets the sync version to 0, so the next update reads the whole library and puts back what is missing. Without `--fulltext` that embeds the whole library again, so `batch-abandon` asks for `--yes` first and says so; in local mode `update-db --fulltext` skips items already indexed, and if the provider still has the rebuild's results, `batch-import` finishes it instead. A batch run submitted before the index was last rebuilt, by either path, was cut against an index that no longer exists: it is never imported automatically and never moves the sync version, though `batch-import --batch-id` still lands its results. If you switch to an embedding model with no Batch API, notes about the old provider's runs are dropped.
- **You're told about a stuck run even if a background update hit it.** The last collect problem and the reason an update waited are shown by `zotero_get_search_database_status` and `db-status`, and as one line on `zotero_semantic_search` results. So are items given up on because their records kept failing to embed, by realtime and batch updates alike: status lists them, and the line stays until they change in Zotero and are written.
- **While a run is holding updates up, the check before a semantic search only collects.** It runs at most hourly instead of on the daily schedule, and never starts a new run. It never replaces the index with a finished force rebuild, because searches running at the time would see a part-filled index: it tells you to run `batch-import`, and `update-db` or the next server start can land it too. With auto-update off it does nothing, so collect by hand with `update-db` or `batch-import`.
- **A failed or expired chunk goes out again, three times at most, and then stops holding the run up.** Rows in any file it left behind are imported, and rows it did not finish go out again in a file of their own. A chunk still failing after that is given up on, so the rest of the run lands. Unless the index still needs `zotero-mcp migrate-index`, its items are kept for a retry: the next updates rewrite them, and after three failed runs they wait until they change in Zotero. On an index that still needs it, the sync version stays put instead, so the next update submits every change since that version again, including chunks of that run already imported. A force rebuild with a chunk given up on writes what arrived over the existing index instead of replacing it.
- **An import doesn't lower the sync version.** The exception is a force rebuild as it lands: it has just replaced the ChromaDB collection, so the sync version has to say what the collection now holds.
- **A force rebuild replaces the index unattended only if it's still current.** Nothing may have synced past its target, no other rebuild may have landed since it was submitted, and its passage chunking must match the index. Nor will it drop documents of other libraries or with no library, if they arrived after you submitted it (a rebuild submitted with `--allow-mass-deletion` doesn't ask again). Otherwise it holds nothing up, and you're told to run `zotero-mcp batch-import`, which replaces the index with it, or `batch-abandon`.

A plain `batch-import` skips runs that a later sync overtook, whose text is older than the index, and runs made with a different embedding model, since no query in the current ChromaDB collection could match their vectors. To import one anyway, name its batches with `--batch-id`.

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
