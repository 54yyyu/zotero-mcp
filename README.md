<!-- mcp-name: io.github.54yyyu/zotero-mcp -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/logo-light.svg" width="96" height="96" alt="Zotero MCP logo">
  </picture>
</p>

<h1 align="center">Zotero MCP: Chat with your Research Library in Claude, ChatGPT, and more</h1>

<p align="center">
  <b>Your Zotero library, local or web, in every AI agent.</b><br>
  Search, read, cite and annotate your papers from Claude, ChatGPT, Codex, Cursor, or a chat panel inside Zotero.
</p>

<p align="center">
  <a href="#quick-start">Install</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#zotero-agent-chat-inside-zotero">Zotero Agent</a> ·
  <a href="https://discord.gg/BvgjbcBUqg">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/zotero-mcp-server/"><img src="https://img.shields.io/pypi/v/zotero-mcp-server?color=cc2936&label=PyPI" alt="PyPI version"></a>
  <a href="https://pepy.tech/projects/zotero-mcp-server"><img src="https://img.shields.io/pepy/dt/zotero-mcp-server?color=cc2936&label=downloads" alt="Downloads"></a>
  <a href="https://discord.gg/BvgjbcBUqg"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/54yyyu/zotero-mcp/blob/main/LICENSE"><img src="https://img.shields.io/github/license/54yyyu/zotero-mcp?color=555" alt="MIT license"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/readme/hero-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/readme/hero-light.webp">
    <img src="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/readme/hero-light.webp" width="900" alt="Zotero with &quot;Attention Is All You Need&quot; open on page 4. The Zotero Agent panel on the right explains why the dot products are scaled by the square root of d_k, with page citations; clicking one has highlighted the cited sentence in the PDF.">
  </picture>
  <br>
  <sub>Zotero Agent explaining a paper with page citations. A click on a citation opens the page and flashes the cited sentence.</sub>
</p>

**Zotero MCP** connects your [Zotero](https://www.zotero.org/) research library with [ChatGPT](https://openai.com), [Claude](https://www.anthropic.com/claude), and other AI assistants (e.g., [Cherry Studio](https://cherry-ai.com/), [Chorus](https://chorus.sh), [Cursor](https://www.cursor.com/)) via the [Model Context Protocol](https://modelcontextprotocol.io/introduction). Search your library, read and annotate papers, add and organize items, and find research by meaning.

> **AI agents:** read [docs/for-agents.md](https://github.com/54yyyu/zotero-mcp/blob/main/docs/for-agents.md) first. It covers which route to use, setup, and the commands in one place.

## Three ways in

One package, `zotero-mcp-server`, ships all three. Pick the one that fits where you work.

| | **MCP server** | **`zotero-cli` + agent skill** | **Zotero Agent** |
|---|---|---|---|
| **For** | Chat apps that speak MCP but have no shell: Claude Desktop, ChatGPT, Cherry Studio, Chorus | Agents with a shell: Claude Code, Codex, Cursor, Windsurf, Gemini CLI, Amp, OpenCode | Chatting inside Zotero, next to the PDF you are reading |
| **How** | `zotero-mcp setup` | `zotero-mcp install-skill` | `zotero-cli plugin`, then install the `.xpi` in Zotero |
| **Why** | Works in any MCP client | 98 tokens in context until it is needed, instead of ~13k | Knows the item, page and selection you have open; answers link to pages |
| **Guide** | [Getting started](https://github.com/54yyyu/zotero-mcp/blob/main/docs/getting-started.md) | [CLI and agent skill](https://github.com/54yyyu/zotero-mcp/blob/main/docs/cli.md) | [Zotero Agent plugin](https://github.com/54yyyu/zotero-mcp/blob/main/docs/chat-plugin.md) |

## What it does

- **Search** by title, author, tag, collection, full text, or meaning ([semantic search](https://github.com/54yyyu/zotero-mcp/blob/main/docs/semantic-search.md) with local, OpenAI, Gemini, or Ollama embeddings)
- **Read** metadata, BibTeX, full text, and page ranges of PDFs, with page images where text extraction garbles math, figures, and tables
- **Annotate**: highlights and area boxes placed on the exact words, figure, table, or equation; notes; PDF annotation extraction
- **Write**: add papers by DOI, URL, ISBN, BibTeX, or file (with open-access PDFs), manage collections and tags, merge duplicates
- **Local or web**: in local mode reads come straight from `zotero.sqlite`; writes go to the running Zotero 10+ or through the web API
- **Scite** citation tallies and retraction alerts (optional)

## Quick start

**1. Install** (Python 3.10+):

```bash
uv tool install zotero-mcp-server     # or: pip install zotero-mcp-server
```

> **New to the command line?** Try the community-built [Zotero MCP Setup](https://github.com/ehawkin/zotero-mcp-setup): a macOS GUI installer, one-click scripts for Mac and Windows, and a step-by-step guide.

**2. Enable Zotero's local API**: in Zotero 7+, open **Settings → Advanced** and tick *Allow other applications on this computer to communicate with Zotero*.

**3. Connect your assistant**:

<details>
<summary><b>Claude Desktop</b></summary>

```bash
zotero-mcp setup      # auto-configures Claude Desktop
```

or add the server by hand to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "zotero": {
      "command": "zotero-mcp",
      "env": { "ZOTERO_LOCAL": "true" }
    }
  }
}
```

</details>

<details>
<summary><b>Claude Code, Codex, Cursor and other agents with a shell</b></summary>

Teach the agent to drive `zotero-cli` (the cheaper route, see [below](#mcp-server-or-agent-skill)):

```bash
zotero-mcp install-skill
```

Or use the MCP server: for Claude Code, add the same `mcpServers` entry as above to `~/.claude.json`.

</details>

<details>
<summary><b>ChatGPT, Cherry Studio, Chorus, Autohand and other clients</b></summary>

See [Getting started](https://github.com/54yyyu/zotero-mcp/blob/main/docs/getting-started.md).

</details>

<details>
<summary><b>Zotero itself (the Zotero Agent chat panel)</b></summary>

```bash
zotero-cli plugin     # where the .xpi is, and how to install it
```

Then in Zotero: **Tools > Plugins**, the gear, **Install Plugin From File**. Details: [Zotero Agent plugin](https://github.com/54yyyu/zotero-mcp/blob/main/docs/chat-plugin.md).

</details>

**4. Writes (optional)**: on Zotero 10+, run `zotero-mcp authorize-local` once and choose **Always Allow**. On older Zotero, add `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID` to write through the web API.

Then ask things like *"Find papers in my library on attention mechanisms"*, *"Summarize the key findings of this paper"*, or *"Highlight the main claims of this PDF"*.

### Optional extras

The base install covers search, reading, annotations, and writes. Heavier features are extras:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `semantic` | Semantic search via ChromaDB, sentence-transformers, OpenAI/Gemini embeddings | `pip install "zotero-mcp-server[semantic]"` |
| `pdf` | PDF outlines, page layout and page images (PyMuPDF), EPUB annotations | `pip install "zotero-mcp-server[pdf]"` |
| `scite` | [Scite](https://scite.ai) citation tallies and retraction alerts (no account needed) | `pip install "zotero-mcp-server[scite]"` |
| `all` | Everything above | `pip install "zotero-mcp-server[all]"` |

Update any time with `zotero-mcp update`.

## MCP server or agent skill?

If your agent has a shell (Claude Code, Cursor, Codex, Windsurf, Gemini CLI, Amp, OpenCode …), one command teaches it to drive `zotero-cli`:

```bash
zotero-mcp install-skill
```

An MCP server sends every tool's schema on every request, before you type anything. The skill costs 98 tokens until the agent decides it is relevant:

| Route | In context | Paid |
|---|---:|---|
| MCP server, default profile (38 tools) | **13,448** | every request |
| Agent skill, frontmatter only | **98** | always |
| Agent skill, body loaded | 1,368 | when it fires |

Use the MCP server when your client speaks MCP but has no shell (Claude Desktop, ChatGPT); use the skill when it has a shell. Both share one config. Details: [CLI and agent skill](https://github.com/54yyyu/zotero-mcp/blob/main/docs/cli.md).

## Zotero Agent: chat inside Zotero

A plugin that adds a chat panel to Zotero itself, driven by your own Claude Code, Codex or pi. It knows which item and page you have open and works your library through `zotero-cli`. Answers cite papers with real `zotero://` links, so a click opens the PDF at that page. It ships in the wheel: run `zotero-cli plugin` for the file and the install steps, or paste the one-paragraph prompt in the docs to your agent and let it do the setup. Details: [Zotero Agent plugin](https://github.com/54yyyu/zotero-mcp/blob/main/docs/chat-plugin.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/readme/library-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/readme/library-light.webp">
    <img src="https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/docs/assets/readme/library-light.webp" width="900" alt="Zotero's library view with a Deep Learning collection. The Zotero Agent panel answers which papers in the library trace how attention replaced recurrence, linking Bahdanau et al. 2015, Vaswani et al. 2017 and Devlin et al. 2019.">
  </picture>
</p>

## Documentation

| Guide | What's in it |
|---|---|
| [Getting started](https://github.com/54yyyu/zotero-mcp/blob/main/docs/getting-started.md) | Connecting Claude Desktop and Claude Code, ChatGPT, Cherry Studio, Chorus, Autohand, and other MCP clients |
| [Configuration](https://github.com/54yyyu/zotero-mcp/blob/main/docs/configuration.md) | Environment variables, local writes, web and hybrid modes, the SQLite read backend, global search, text extraction, command-line options |
| [Semantic search](https://github.com/54yyyu/zotero-mcp/blob/main/docs/semantic-search.md) | Embedding models, building and updating the index |
| [Tools](https://github.com/54yyyu/zotero-mcp/blob/main/docs/tools.md) | Every MCP tool, tool groups (`ZOTERO_MCP_TOOLSETS`), related items, PDF annotation extraction |
| [CLI and agent skill](https://github.com/54yyyu/zotero-mcp/blob/main/docs/cli.md) | `zotero-cli` command reference, `--json` output, `install-skill` |
| [Zotero Agent plugin](https://github.com/54yyyu/zotero-mcp/blob/main/docs/chat-plugin.md) | The chat panel inside Zotero: install, first run, agents and sign-in, privacy |
| [Docker](https://github.com/54yyyu/zotero-mcp/blob/main/docs/docker-images.md) | Container images and runtime modes |
| [Troubleshooting](https://github.com/54yyyu/zotero-mcp/blob/main/docs/troubleshooting.md) | Common problems and fixes |
| [For AI agents](https://github.com/54yyyu/zotero-mcp/blob/main/docs/for-agents.md) | One guide for an agent setting up or using Zotero MCP |

Website: [stevenyuyy.com/zotero-mcp](https://stevenyuyy.com/zotero-mcp/) · [Changelog](https://github.com/54yyyu/zotero-mcp/blob/main/CHANGELOG.md)

## Contributing

Issues and pull requests are welcome. Run the tests with `uv run pytest tests/`. A live integration test plan, meant to be run by Claude against a real library, is in [docs/integration-test-plan.md](https://github.com/54yyyu/zotero-mcp/blob/main/docs/integration-test-plan.md).

Thanks to everyone who has contributed code, fixes, and ideas to Zotero MCP.

<p align="center">
  <a href="https://github.com/54yyyu/zotero-mcp/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=54yyyu/zotero-mcp&max=120&columns=18" width="720" alt="Contributors to Zotero MCP">
  </a>
</p>

<p align="center">
  <a href="https://star-history.com/#54yyyu/zotero-mcp&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=54yyyu/zotero-mcp&type=Date&theme=dark">
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=54yyyu/zotero-mcp&type=Date">
      <img src="https://api.star-history.com/svg?repos=54yyyu/zotero-mcp&type=Date" width="600" alt="Star history of 54yyyu/zotero-mcp">
    </picture>
  </a>
</p>

## Support

Zotero MCP is free and MIT-licensed.

If it saves you or your lab time, sponsoring helps cover the unglamorous parts: Windows and WSL2 edge
cases, Zotero schema changes, group-library support, and the embedding/search infrastructure.

<a href="https://github.com/sponsors/54yyyu"><img src="https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa?logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub"></a> <a href="https://buymeacoffee.com/stevenyuyy"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>

**Labs and institutions:** the $50 and $200 tiers are meant to be expensable, and include priority
triage on the issues affecting your workflow.

## License

MIT
