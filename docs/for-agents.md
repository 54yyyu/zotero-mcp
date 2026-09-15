# Zotero MCP for AI agents

You are an AI agent that has found Zotero MCP, probably because a user wants you to work with their Zotero research library, or to set that up. This page is the single place to learn how. It is written for you; the [README](../README.md) is written for people.

## What it is

Zotero MCP connects a [Zotero](https://www.zotero.org/) library to AI assistants. It ships two entry points from one Python package, `zotero-mcp-server`:

| Entry point | What it is | Use it when |
|---|---|---|
| `zotero-cli` | A command-line client with `--json` output on every command | **You can run shell commands.** It costs no context until used. |
| `zotero-mcp serve` | An MCP server exposing ~38 tools by default | Your client speaks MCP but has no shell (Claude Desktop, ChatGPT). |

Both read the same configuration. If you have a shell, prefer `zotero-cli`: the MCP server's tool schemas cost about 13.4k tokens on every request whether or not they are used.

## Setting it up

Check before installing; it may already be there:

```bash
zotero-cli config                  # prints resolved settings, or fails if not installed/configured
```

If not installed:

```bash
uv tool install zotero-mcp-server  # or: pip install zotero-mcp-server
zotero-mcp setup                   # interactive; configures Claude Desktop if present
zotero-mcp install-skill           # teaches coding agents in this project to use zotero-cli
```

Optional extras: `[pdf]` for page layout, page images, outlines and EPUB annotations; `[semantic]` for search by meaning; `[scite]` for citation tallies; `[all]` for everything.

### Connection modes

| Mode | What the user must do | Reads | Writes |
|---|---|---|---|
| Local | Zotero desktop running, with *Settings → Advanced → Allow other applications on this computer to communicate with Zotero* ticked; `ZOTERO_LOCAL=true` | ✓ (straight from `zotero.sqlite`) | ✗ |
| Local + write (Zotero 10+) | Also run `zotero-mcp authorize-local` once and click **Always Allow** in the dialog Zotero shows | ✓ | ✓ to the running Zotero |
| Hybrid (older Zotero) | Local, plus `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID` | ✓ local | ✓ via web API |
| Web | `ZOTERO_API_KEY`, `ZOTERO_LIBRARY_ID` (and `ZOTERO_LIBRARY_TYPE=group` for a group library) | ✓ | ✓ |

`authorize-local` needs a human to click in Zotero. Tell the user to expect the dialog; do not retry it in a loop (Zotero rate-limits it). An API key comes from https://www.zotero.org/settings/security, and the numeric user ID is on the same page.

For an MCP client without `zotero-mcp setup` support, the server entry is:

```json
{ "mcpServers": { "zotero": { "command": "zotero-mcp", "env": { "ZOTERO_LOCAL": "true" } } } }
```

If a GUI client cannot find `zotero-mcp`, use the absolute path from `zotero-mcp setup-info`. Per-client instructions: [Getting started](getting-started.md). Every setting: [Configuration](configuration.md).

## Using zotero-cli

Always pass `--json` when you will read the output. Every command prints one object to stdout:

```json
{"ok": true, "command": "search", "schema": 1, "data": {...}}
{"ok": false, "command": "search", "schema": 1, "error": {"message": "...", "code": "..."}}
```

Check `ok` before using `data`; a failed command also exits 1. `zotero-cli --json-schema` prints the full contract, and `zotero-cli <command> --help` works for any command.

**Item keys are the currency.** Almost every task is: find keys, then act on them.

```bash
zotero-cli --json search "attention mechanisms" --limit 10 --detail keys_only
zotero-cli --json get metadata ABCD1234
zotero-cli --json get children ABCD1234          # attachment keys (PDFs) and notes
```

Search modes: `items` (default: title, author, phrase), `semantic` (meaning; needs the index), `tag`, `advanced`, `citekey`, `notes`. Add `--all-libraries` to search group libraries too (local mode).

### Reading a paper

```bash
zotero-cli --json outline ITEM_KEY                              # find the section
zotero-cli --json read ITEM_KEY --start-page 3 --end-page 8     # read only those pages
```

Extracted text is reliable for prose and **unreliable for math, figures and tables**: symbols drop out and table cells run together. `read` ends each page with a note such as `Garbled in this text: Equation (1), Table 2`. For those pages, look at the page itself if you can view images:

```bash
zotero-cli --json read ITEM_KEY --start-page 4 --format image                               # PNG paths
zotero-cli --json read ITEM_KEY --start-page 4 --format image --rect 0.35,0.49,0.3,0.05     # zoom in
zotero-cli path ITEM_KEY                                                                    # the PDF on disk
```

### Annotating a paper

Plan everything, check placement, then write in one run:

1. `zotero-cli --json layout ATTACHMENT_KEY --pages 3-9` lists figure, table and equation boxes with captions and a ready `rect_arg`.
2. Write a JSON Lines plan: `{"page": 4, "text": "exact words", "comment": "...", "color": "yellow"}` for a highlight, `{"page": 3, "rect": "x,y,w,h", "comment": "..."}` for a box. Copy highlight text exactly from `read`; it is searched on that page and two either side.
3. `zotero-cli annotations batch --attachment-key ATTACHMENT_KEY --file plan.jsonl --dry-run` prints the words each highlight would cover. Fix every miss.
4. Run it again without `--dry-run`. Failures are listed with `ok: false` and the exit code is 1.

Colors take Zotero's names (yellow, red, green, blue, purple, magenta, orange, gray). Three or four colors with fixed meanings read better than eight; explain them in a note (`zotero-cli notes create --item-key ITEM_KEY --text -`).

### Writing

```bash
zotero-cli add doi 10.1038/s41586-021-03819-2 -c "Reading List"   # idempotent: re-running files, never duplicates
zotero-cli edit ABCD1234 --add-tags reviewed
zotero-cli batch --item-keys A1B2C3D4,E5F6G7H8 --add-tags screened
```

The full command reference is [docs/cli.md](cli.md) and, generated from the parser, [reference.md](../src/zotero_mcp/skills/zotero-cli/reference.md).

## Using the MCP server

The tools mirror the CLI: `zotero_search_items`, `zotero_get_item_metadata`, `zotero_get_item_children`, `zotero_read_pdf_pages` (with `format='image'` and `rect`), `zotero_get_page_layout`, `zotero_create_annotation` (`text=` for a highlight, `rect=` for a box), `zotero_manage_note`, `zotero_add_by_doi`, and more. Some groups are opt-in through `ZOTERO_MCP_TOOLSETS`. Every tool and group: [Tools](tools.md).

## Rules that prevent mistakes

- **Confirm before destructive changes** (deleting items or annotations, merging duplicates, batch edits over many items) and show the user what will change. Annotation deletes are permanent.
- **Do not report "no papers on X" from an empty result** until you know the library is reachable (`zotero-cli config`) and, for semantic search, indexed (`zotero-cli db status`; build with `zotero-cli db update`).
- **A write refused with "Cannot perform write operations"** means no write path is configured. The message names both fixes (`zotero-mcp authorize-local`, or web API credentials); relay it rather than retrying.
- **An item marked `"deleted": true` is in the trash**; do not treat it as part of the live library.
- **A partial-results note** on a search means the scan was cut short, not that nothing else matched. Narrow the query.
- **The user's library may sync to zotero.org.** Everything you write reaches their other devices.

More help: [Troubleshooting](troubleshooting.md).
