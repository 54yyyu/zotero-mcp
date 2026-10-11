# Web discovery alongside your Zotero library

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to find research on the web and fetch page excerpts alongside Zotero's library
tools. Anonymous search uses Fast mode and needs no Parallel account or API key.
The free endpoint has rate limits.

This opt-in example uses FastMCP's multi-server client: Zotero runs over stdio,
and Parallel uses Streamable HTTP. It does not modify your saved MCP configuration.

## Run the example

From the repository root, with Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv-parallel
uv pip install --python .venv-parallel . "fastmcp>=2.14,<4"
# macOS/Linux:
source .venv-parallel/bin/activate
# Windows PowerShell instead:
# .venv-parallel\Scripts\Activate.ps1
python examples/parallel-search/search.py "attention mechanisms original paper"
python examples/parallel-search/search.py "attention mechanisms original paper" --fetch https://arxiv.org/abs/1706.03762
```

The script loads the adjacent `mcp.json`, starts the installed `zotero-mcp`, lists
both servers' tools, and prints search results with URLs and excerpts as JSON.
`--fetch` also returns page excerpts for the supplied URL. Each invocation shares
one session identifier across its search and fetch calls. Connection initialization
has a 30-second timeout and tool calls have a 60-second timeout; MCP failures exit
with status 1. A fetch response can contain per-URL `errors`, so inspect those as
well as `results`.

Zotero tool discovery does not require a running library. To use the library tools,
follow [Getting started](../../docs/getting-started.md) to configure local Zotero.
The example's Zotero entry uses `ZOTERO_LOCAL=true`; for web or hybrid access,
adapt only that entry using your existing Zotero settings. If `zotero-mcp` is not
on your PATH, set `command` to its absolute path.

## Continue the research workflow

FastMCP prefixes tool names with the server name: Parallel exposes
`parallel_web_search` and `parallel_web_fetch`; Zotero exposes names such as
`zotero_zotero_search_items` and `zotero_zotero_add_by_doi`. Use `available_tools`
to see the enabled tools. In your own FastMCP client, use `list_tools()` to inspect
each tool's `inputSchema` before calling it.

Search the web for candidate papers, compare their titles or DOIs with your
library using Zotero's search tools, then explicitly add a selected paper using
Zotero's DOI or URL import tools once writes are configured. The script performs
web search and optional fetching only; it does not call library tools, import
papers, or run a model. Web results are source leads, not verified bibliographic
records. Check the paper's DOI and publication details before importing it.

The `parallel` entry sends no authentication header and includes a project
User-Agent. Keep credentials in your Zotero entry; Parallel's public endpoint
does not need them. Queries and requested page URLs are sent to Parallel.
