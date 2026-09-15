# CLI and agent skill

`zotero-cli` is a standalone terminal interface to your Zotero library. It uses the same tools as the MCP server but without needing an AI assistant — useful for quick lookups, shell scripts, and automation.

Use `zotero-mcp` when your AI client supports MCP (Claude Desktop, ChatGPT). Use `zotero-cli` for shell scripts, cron jobs, or agentic pipelines with shell access (e.g. Claude Code) — CLI commands cost far fewer tokens than MCP tool schemas and compose naturally with Unix pipes.

Both share the same configuration set up by `zotero-mcp setup`. Short aliases (`s`, `g`, `ann`, `coll`) are there for interactive use.

## How much context each route costs

The MCP server sends every enabled tool's name, description and JSON parameter schema to the model on **every request**, before you type anything. The CLI route puts only a skill description in context until the model decides it is relevant. Measured on this repo with `python scripts/measure_context_cost.py`:

| Route | Tokens in context | When it is paid |
|---|---:|---|
| MCP, default profile (38 tools) | 13,448 | every request |
| MCP, `ZOTERO_MCP_TOOLSETS=none` (32 tools) | 11,761 | every request |
| MCP, `ZOTERO_MCP_TOOLSETS=all` (50 tools) | 17,414 | every request |
| CLI skill, frontmatter only | 98 | always |
| CLI skill, body loaded | 1,368 | once the skill fires |
| CLI skill + full command reference | 4,389 | worst case |

That is the *fixed* cost only. It does not measure task success, output size, or how many round trips each route takes to finish a job — a cheaper surface that gets the answer wrong is not cheaper. Numbers are `cl100k_base` tokens and are re-measured, not estimated; `tests/test_context_cost_claim.py` fails if the relationship stops holding.

## 🪶 Agent skill: one command for any harness

```bash
zotero-mcp install-skill
```

Run it in your project. It detects which agent harnesses are set up there and installs to each one, in that harness's own format:

| Harness | Detected by | Installs |
|---|---|---|
| Claude Code (project) | `.claude/` | `.claude/skills/zotero-cli/` |
| Claude Code (user) | `~/.claude/` | `~/.claude/skills/zotero-cli/` |
| Cursor | `.cursor/` | `.cursor/rules/zotero-cli.mdc` |
| Windsurf | `.windsurf/` | `.windsurf/rules/zotero-cli.md` |
| Codex, Amp, OpenCode, Jules … | `AGENTS.md` | a pointer block in `AGENTS.md` |
| Gemini CLI | `GEMINI.md` or `.gemini/` | a pointer block in `GEMINI.md` |

```bash
zotero-mcp install-skill --list-targets      # what is detected here
zotero-mcp install-skill --target cursor     # install one explicitly
zotero-mcp install-skill --force             # overwrite an existing copy
```

**It will not overwrite your work.** A destination that exists and differs is reported, not replaced, unless you pass `--force`. For shared instruction files it is stricter: only the text between the `zotero-cli` markers is ever managed, so the rest of your `AGENTS.md` is untouchable by construction — re-running updates that block in place rather than appending a second one.

**It keeps the context advantage.** Shared instruction files get a short pointer block, not the whole skill; the body lands beside it and the agent opens it only when it decides Zotero is relevant. Pasting 1,400 tokens into every agent's always-loaded context would spend exactly the advantage this exists for.

The skill teaches the find-keys-then-act loop, `--json`, how to pick among the six search modes, paging, reading a PDF by outline-then-page-range rather than whole, reading and annotating a paper, and when an empty result means "the index is not built" rather than "you have no papers on that". Its generated [command reference](../src/zotero_mcp/skills/zotero-cli/reference.md) lists every command and flag.

## Machine-readable output (`--json`)

Every command accepts `--json`, before or after the command name. Output is one object per invocation:

```bash
zotero-cli --json search "attention" --limit 5 --detail keys_only
# {"ok": true, "command": "search", "schema": 1, "data": {"count": 5, "items": [...]}}
```

Success carries `data`; failure carries `error.message` and a stable `error.code`, also on stdout, so one stream carries both outcomes, and a failed command exits 1. Read commands (search, get, annotations list, notes list, config) return real structure; commands whose answer is a status line return `{"text": ...}`. Run `zotero-cli --json-schema` for the full contract.

```bash
# Item keys are the currency of every command — pipe them onward
zotero-cli --json search "diffusion models" --limit 5 --detail keys_only \
  | jq -r '.data.items[].key' \
  | while read -r key; do zotero-cli --json get metadata "$key"; done
```

## Quick reference

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
zotero-cli ann list --item-key ABC123         # annotations (short alias)
zotero-cli ann list --item-key ABC123 --format json  # structured export
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

# Reading PDFs — find the section first, then read only those pages
zotero-cli outline ABC123
zotero-cli read ABC123 --start-page 42 --end-page 55      # flags garbled math, figures, tables
zotero-cli read ABC123 --start-page 44 --format image    # PNG page images (up to 10 pages)
zotero-cli read ABC123 --start-page 44 --format image --rect 0.35,0.49,0.3,0.05   # zoom in
zotero-cli path ABC123                        # where the file lives on disk

# Annotating a PDF — boxes to aim at, then a checked plan written in one run
zotero-cli --json layout ATTACH01 --pages 3-9            # figure, table and equation boxes
zotero-cli annotations create --attachment-key ATTACH01 --page 4 --text "exact words" --color yellow
zotero-cli annotations create --attachment-key ATTACH01 --page 3 --rect 0.32,0.09,0.36,0.41 --comment "Figure 1"
zotero-cli annotations batch --attachment-key ATTACH01 --file plan.jsonl --dry-run   # where each highlight lands
zotero-cli annotations batch --attachment-key ATTACH01 --file plan.jsonl             # write them all

# Attachments, deletion, bibliographies
zotero-cli attach ABC123 --file /path/to/paper.pdf
zotero-cli delete item ABC123
zotero-cli export --item-keys ABC123,DEF456 --style apa
zotero-cli export --collection COLL01 --format bibtex

# Discovery and synthesis
zotero-cli related 10.1038/s41586-021-03819-2 --direction citations
zotero-cli coverage --collection COLL01
zotero-cli synthesize --tag "to-read" --format json

# Bulk edits across many items
zotero-cli batch --item-keys ABC123,DEF456 --add-tags screened
zotero-cli batch --query "machine learning" --add-tags survey --limit 100
```

An annotation plan (`plan.jsonl`) is one JSON object per line: `{"page": 4, "text": "exact words", "comment": "...", "color": "yellow"}` for a highlight, or `{"page": 3, "rect": "x,y,w,h", "comment": "..."}` for a box. Colors take Zotero's names: yellow, red, green, blue, purple, magenta, orange, gray.

Paging: listings cap at `--limit` and the response names the next offset.

```bash
zotero-cli --json get collection-items QS7TQPPA --limit 100 --offset 100
```

## Verbose mode

Add `-v` anywhere to see progress messages (e.g., which API calls are made):

```bash
zotero-cli -v search "CRISPR"
```
