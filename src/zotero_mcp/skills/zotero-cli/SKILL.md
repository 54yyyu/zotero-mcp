---
name: zotero-cli
description: Read and write a Zotero library from the shell with the `zotero-cli` command - search papers by keyword or meaning, read PDF full text and page ranges, get and set metadata, manage collections, tags, notes and annotations, add items by DOI/URL/ISBN, and export bibliographies. Use whenever the user asks about their Zotero library, references, citations, papers they have saved, or their reading notes.
---

# Zotero from the shell

`zotero-cli` reaches a Zotero library directly. Prefer it over the Zotero MCP
server when you have shell access: the MCP server's tool schemas cost ~14k
tokens of context on every request whether or not you use them, while this
costs only what you actually run.

Check it is set up before the first real call:

```bash
zotero-cli config          # prints the resolved Zotero settings
```

If that fails, Zotero is not reachable. Local mode needs the Zotero desktop
app running with its local API enabled; web mode needs `ZOTERO_API_KEY` and
`ZOTERO_LIBRARY_ID`. Say so rather than guessing at library contents.

## Always pass --json when you will read the output

Default output is markdown for humans. `--json` gives you one object per
invocation with a stable shape, so you never parse prose:

```json
{"ok": true, "command": "search", "schema": 1, "data": {...}}
{"ok": false, "command": "search", "schema": 1, "error": {"message": "...", "code": "..."}}
```

Both go to **stdout**; `[INFO]`/`[WARN]` diagnostics go to stderr. Check `ok`
before using `data`. Run `zotero-cli --json-schema` for the full contract.

## The core loop

Almost every task is: **find keys → act on keys**. Item keys are 8
characters and are the currency of every command.

```bash
# 1. find - use --detail keys_only to keep the result small while browsing
zotero-cli --json search "attention mechanisms" --limit 10 --detail keys_only

# 2. act - metadata, full text, or a page range
zotero-cli --json get metadata ABCD1234
zotero-cli --json get fulltext ABCD1234
zotero-cli --json read ABCD1234 --start-page 3 --end-page 8
```

Pipe keys straight into the next call:

```bash
zotero-cli --json search "diffusion models" --limit 5 --detail keys_only \
  | jq -r '.data.items[].key' \
  | while read -r key; do zotero-cli --json get metadata "$key"; done
```

## Choosing a search mode

| Mode | Use it for | Command |
|---|---|---|
| `items` (default) | a title, author or phrase you know | `search "Vaswani attention"` |
| `semantic` | a topic or idea, no exact wording | `search --mode semantic "why transformers scale"` |
| `tag` | items you filed under a tag | `search --mode tag "to-read,important"` |
| `advanced` | structured field conditions | `search --mode advanced --conditions '[...]'` |
| `citekey` | a BibTeX citation key | `search --mode citekey smith2020` |
| `notes` | text inside your notes, not item fields | `search --mode notes "research question"` |

`semantic` needs the search index built (`zotero-cli db status` to check,
`zotero-cli db update` to build). If it is empty, fall back to `items` and
say why rather than reporting no results.

Every search covers one library — the active one. When you don't know which
library holds something, or a search comes up empty and the item might live in
a group library, add `--all-libraries`:

```bash
zotero-cli search "Cladder-Micus" --all-libraries
```

Each result is then labelled `**Library:** <name>`. It needs the SQLite backend,
the default in local mode, and errors clearly if it is not in use, so try it
once and fall back to per-library searches if it is refused. Tag filters work
with it; `--collection` does not, because a collection lives inside one library.

## Reading efficiently

`get fulltext` on a book-length PDF returns a great deal of text. When you
need one section, use the outline to find it and read only those pages:

```bash
zotero-cli --json outline ABCD1234
zotero-cli --json read ABCD1234 --start-page 42 --end-page 55
```

For "what does my library say about X", prefer `search --mode semantic`
followed by targeted `get metadata` over reading whole papers.

## Paging

Listings cap at `--limit`. When more exists, the response says so and names
the next offset:

```bash
zotero-cli --json get collection-items QS7TQPPA --limit 100 --offset 100
```

Keep going until `data.count` is less than `--limit`.

## Writing

Write commands report what they did as text under `data.text`.

```bash
zotero-cli add doi 10.1038/s41586-021-03819-2 -c "Reading List"
zotero-cli edit ABCD1234 --title "Corrected Title" --add-tags reviewed
zotero-cli notes create --item-key ABCD1234 --text "Key finding: ..."
zotero-cli batch --item-keys A1B2C3D4,E5F6G7H8 --add-tags screened
```

`add` is idempotent by default: re-running files the existing item into the
named collection rather than creating a duplicate. Use `--if-exists skip` to
never touch an existing item.

Before a destructive change (`delete`, `duplicates merge`, a `batch` over
many items), confirm with the user and show what will be affected. `delete
item` refuses notes unless `--allow-note` is passed.

## Reading and annotating a paper

```bash
zotero-cli get children ITEM_KEY                          # the PDF's attachment key
zotero-cli read ITEM_KEY --start-page 1 --end-page 99     # end page clamps to the last page
zotero-cli path ATTACHMENT_KEY                            # the PDF file on disk
```

Extracted text is reliable for prose and unreliable for math, figures and
tables: symbols drop out and table cells run together. `read` flags each
page where that happens ("Garbled in this text: Equation (1), Table 2").
For those pages, look at the page itself if you can view images:

```bash
zotero-cli read ITEM_KEY --start-page 4 --format image                         # PNG per page, up to 10
zotero-cli read ITEM_KEY --start-page 4 --format image --rect 0.35,0.49,0.3,0.05   # zoom into one region
```

To annotate, plan everything, check it, then write it in one run:

1. `zotero-cli --json layout ATTACHMENT_KEY` lists figure, table and
   equation boxes with their captions and a paste-ready `rect_arg`.
2. Write one JSON object per line: `{"page": 4, "text": "exact words",
   "comment": "...", "color": "yellow"}` for a highlight, or
   `{"page": 3, "rect": "x,y,w,h", "comment": "..."}` for a box. Copy
   highlight text exactly from `read`; it is searched for on that page and
   two pages either side.
3. `zotero-cli annotations batch --attachment-key ATTACHMENT_KEY --file plan.jsonl --dry-run`
   shows the words each highlight would cover. Fix every miss.
4. Run it again without `--dry-run`. Anything that did not land is listed
   under `data.results` with `ok: false`, and the exit code is 1.

Colors take Zotero's names: yellow, red, green, blue, purple, magenta,
orange, gray. Three or four colors with fixed meanings read better than
eight; say what they mean in a note on the item.

Writes in local mode need a one-time `zotero-mcp authorize-local` (Zotero 10
or newer) or web API credentials. A write refused for that reason says so.

## When something looks wrong

- Empty search results: check `zotero-cli config` and, for semantic mode,
  `zotero-cli db status`. Do not report "you have no papers on X" until you
  know the library is actually reachable and indexed.
- `ok: false`: read `error.message`. It names the cause.
- A partial-results note on a search means the scan was cut short, not that
  nothing else matched. Narrow the query and re-run.
- An item marked `"deleted": true` is in the trash. Do not treat it as part
  of the live library.

## Full command reference

`reference.md` in this skill directory lists every command and flag. Read it
when you need something not covered above. `zotero-cli <command> --help`
also works for any single command.
