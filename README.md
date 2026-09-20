<!-- mcp-name: io.github.54yyyu/zotero-mcp -->

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

**Zotero MCP** connects your [Zotero](https://www.zotero.org/) research library with [ChatGPT](https://openai.com), [Claude](https://www.anthropic.com/claude), and other AI assistants (e.g., [Cherry Studio](https://cherry-ai.com/), [Chorus](https://chorus.sh), [Cursor](https://www.cursor.com/)) via the [Model Context Protocol](https://modelcontextprotocol.io/introduction). Search your library, read and annotate papers, add and organize items, and find research by meaning.

> **AI agents:** read [docs/for-agents.md](https://github.com/54yyyu/zotero-mcp/blob/main/docs/for-agents.md) first. It covers which route to use, setup, and the commands in one place.

## ✨ What it does

- 🔍 **Search** by title, author, tag, collection, full text, or meaning ([semantic search](https://github.com/54yyyu/zotero-mcp/blob/main/docs/semantic-search.md) with local, OpenAI, Gemini, or Ollama embeddings)
- 📚 **Read** metadata, BibTeX, full text, and page ranges of PDFs, with page images where text extraction garbles math, figures, and tables
- 📝 **Annotate**: highlights and area boxes placed on the exact words, figure, table, or equation; notes; PDF annotation extraction
- ✏️ **Write**: add papers by DOI, URL, ISBN, BibTeX, or file (with open-access PDFs), manage collections and tags, merge duplicates
- 💻 **Local or web**: in local mode reads come straight from `zotero.sqlite`; writes go to the running Zotero 10+ or through the web API
- 🪶 **Two ways in**: an MCP server for chat apps, or `zotero-cli` plus an agent skill for coding agents
- 📊 **Scite** citation tallies and retraction alerts (optional)

## 🚀 Quick start

**1. Install** (Python 3.10+):

```bash
uv tool install zotero-mcp-server     # or: pip install zotero-mcp-server
```

> **New to the command line?** Try the community-built [Zotero MCP Setup](https://github.com/ehawkin/zotero-mcp-setup): a macOS GUI installer, one-click scripts for Mac and Windows, and a step-by-step guide.

**2. Enable Zotero's local API**: in Zotero 7+, open **Settings → Advanced** and tick *Allow other applications on this computer to communicate with Zotero*.

**3. Connect your assistant**:

```bash
zotero-mcp setup      # auto-configures Claude Desktop
```

or add the server by hand (Claude Desktop: `claude_desktop_config.json`; Claude Code: `~/.claude.json`):

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

**4. Writes (optional)**: on Zotero 10+, run `zotero-mcp authorize-local` once and choose **Always Allow**. On older Zotero, add `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID` to write through the web API.

Then ask things like *"Find papers in my library on attention mechanisms"*, *"Summarize the key findings of this paper"*, or *"Highlight the main claims of this PDF"*.

ChatGPT, Cherry Studio, Chorus, Autohand, and other clients: see [Getting started](https://github.com/54yyyu/zotero-mcp/blob/main/docs/getting-started.md).

### Optional extras

The base install covers search, reading, annotations, and writes. Heavier features are extras:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `semantic` | Semantic search via ChromaDB, sentence-transformers, OpenAI/Gemini embeddings | `pip install "zotero-mcp-server[semantic]"` |
| `pdf` | PDF outlines, page layout and page images (PyMuPDF), EPUB annotations | `pip install "zotero-mcp-server[pdf]"` |
| `scite` | [Scite](https://scite.ai) citation tallies and retraction alerts (no account needed) | `pip install "zotero-mcp-server[scite]"` |
| `all` | Everything above | `pip install "zotero-mcp-server[all]"` |

Update any time with `zotero-mcp update`.

## 🪶 MCP server or agent skill?

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

## 📖 Documentation

| Guide | What's in it |
|---|---|
| [Getting started](https://github.com/54yyyu/zotero-mcp/blob/main/docs/getting-started.md) | Connecting Claude Desktop and Claude Code, ChatGPT, Cherry Studio, Chorus, Autohand, and other MCP clients |
| [Configuration](https://github.com/54yyyu/zotero-mcp/blob/main/docs/configuration.md) | Environment variables, local writes, web and hybrid modes, the SQLite read backend, global search, text extraction, command-line options |
| [Semantic search](https://github.com/54yyyu/zotero-mcp/blob/main/docs/semantic-search.md) | Embedding models, building and updating the index |
| [Tools](https://github.com/54yyyu/zotero-mcp/blob/main/docs/tools.md) | Every MCP tool, tool groups (`ZOTERO_MCP_TOOLSETS`), related items, PDF annotation extraction |
| [CLI and agent skill](https://github.com/54yyyu/zotero-mcp/blob/main/docs/cli.md) | `zotero-cli` command reference, `--json` output, `install-skill` |
| [Docker](https://github.com/54yyyu/zotero-mcp/blob/main/docs/docker-images.md) | Container images and runtime modes |
| [Troubleshooting](https://github.com/54yyyu/zotero-mcp/blob/main/docs/troubleshooting.md) | Common problems and fixes |
| [For AI agents](https://github.com/54yyyu/zotero-mcp/blob/main/docs/for-agents.md) | One guide for an agent setting up or using Zotero MCP |

Website: [stevenyuyy.com/zotero-mcp](https://stevenyuyy.com/zotero-mcp/) · [Changelog](https://github.com/54yyyu/zotero-mcp/blob/main/CHANGELOG.md)

## 🤝 Contributing

Issues and pull requests are welcome. Run the tests with `uv run pytest tests/`. Where code goes and why — the package map, naming rules, test layout and import-cost budget — is in [docs/architecture.md](https://github.com/54yyyu/zotero-mcp/blob/main/docs/architecture.md). A live integration test plan, meant to be run by Claude against a real library, is in [docs/integration-test-plan.md](https://github.com/54yyyu/zotero-mcp/blob/main/docs/integration-test-plan.md).

## ☕ Support

Zotero MCP is free and MIT-licensed.

If it saves you or your lab time, sponsoring helps cover the unglamorous parts: Windows and WSL2 edge
cases, Zotero schema changes, group-library support, and the embedding/search infrastructure.

<a href="https://github.com/sponsors/54yyyu">
  <img src="https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub">
</a>
<a href="https://buymeacoffee.com/stevenyuyy">
  <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">
</a>

**Labs and institutions:** the $50 and $200 tiers are meant to be expensable, and include priority
triage on the issues affecting your workflow.

## Contributors

Thanks to everyone who has contributed code, fixes, and ideas to Zotero MCP.

<!-- contributors:start -->
<p align="center">
<a href="https://github.com/54yyyu" title="54yyyu"><img src="https://avatars.githubusercontent.com/u/27888654?v=4&s=64" width="32" height="32" alt="54yyyu"></a>
<a href="https://github.com/mronkko" title="mronkko"><img src="https://avatars.githubusercontent.com/u/566094?v=4&s=64" width="32" height="32" alt="mronkko"></a>
<a href="https://github.com/josk0" title="josk0"><img src="https://avatars.githubusercontent.com/u/53160902?v=4&s=64" width="32" height="32" alt="josk0"></a>
<a href="https://github.com/danmackinlay" title="danmackinlay"><img src="https://avatars.githubusercontent.com/u/21740?v=4&s=64" width="32" height="32" alt="danmackinlay"></a>
<a href="https://github.com/peterdresslar" title="peterdresslar"><img src="https://avatars.githubusercontent.com/u/2095978?v=4&s=64" width="32" height="32" alt="peterdresslar"></a>
<a href="https://github.com/rbpasker" title="rbpasker"><img src="https://avatars.githubusercontent.com/u/87866?v=4&s=64" width="32" height="32" alt="rbpasker"></a>
<a href="https://github.com/ehawkin" title="ehawkin"><img src="https://avatars.githubusercontent.com/u/74272474?v=4&s=64" width="32" height="32" alt="ehawkin"></a>
<a href="https://github.com/brianckeegan" title="brianckeegan"><img src="https://avatars.githubusercontent.com/u/1253373?v=4&s=64" width="32" height="32" alt="brianckeegan"></a>
<a href="https://github.com/StStME" title="StStME"><img src="https://avatars.githubusercontent.com/u/33428955?v=4&s=64" width="32" height="32" alt="StStME"></a>
<a href="https://github.com/davidszp" title="davidszp"><img src="https://avatars.githubusercontent.com/u/15107452?v=4&s=64" width="32" height="32" alt="davidszp"></a>
<a href="https://github.com/lots-o" title="lots-o"><img src="https://avatars.githubusercontent.com/u/39071632?v=4&s=64" width="32" height="32" alt="lots-o"></a>
<a href="https://github.com/EwoutH" title="EwoutH"><img src="https://avatars.githubusercontent.com/u/15776622?v=4&s=64" width="32" height="32" alt="EwoutH"></a>
<a href="https://github.com/QuentinAndre" title="QuentinAndre"><img src="https://avatars.githubusercontent.com/u/8501935?v=4&s=64" width="32" height="32" alt="QuentinAndre"></a>
<a href="https://github.com/kubanowalski" title="kubanowalski"><img src="https://avatars.githubusercontent.com/u/61462204?v=4&s=64" width="32" height="32" alt="kubanowalski"></a>
<a href="https://github.com/ajdavis" title="ajdavis"><img src="https://avatars.githubusercontent.com/u/84101?v=4&s=64" width="32" height="32" alt="ajdavis"></a>
<a href="https://github.com/jiahaoh" title="jiahaoh"><img src="https://avatars.githubusercontent.com/u/33877832?v=4&s=64" width="32" height="32" alt="jiahaoh"></a>
<a href="https://github.com/xxie-xd" title="xxie-xd"><img src="https://avatars.githubusercontent.com/u/85358920?v=4&s=64" width="32" height="32" alt="xxie-xd"></a>
<a href="https://github.com/trahloff" title="trahloff"><img src="https://avatars.githubusercontent.com/u/16914641?v=4&s=64" width="32" height="32" alt="trahloff"></a>
<a href="https://github.com/ZhenhongDu" title="ZhenhongDu"><img src="https://avatars.githubusercontent.com/u/61380549?v=4&s=64" width="32" height="32" alt="ZhenhongDu"></a>
<a href="https://github.com/ahharvey" title="ahharvey"><img src="https://avatars.githubusercontent.com/u/678140?v=4&s=64" width="32" height="32" alt="ahharvey"></a>
<a href="https://github.com/take0x" title="take0x"><img src="https://avatars.githubusercontent.com/u/89313929?v=4&s=64" width="32" height="32" alt="take0x"></a>
<a href="https://github.com/linozen" title="linozen"><img src="https://avatars.githubusercontent.com/u/37184648?v=4&s=64" width="32" height="32" alt="linozen"></a>
<a href="https://github.com/raffaelemancuso" title="raffaelemancuso"><img src="https://avatars.githubusercontent.com/u/54762742?v=4&s=64" width="32" height="32" alt="raffaelemancuso"></a>
<a href="https://github.com/michaelzehetleitner" title="michaelzehetleitner"><img src="https://avatars.githubusercontent.com/u/90147439?v=4&s=64" width="32" height="32" alt="michaelzehetleitner"></a>
<a href="https://github.com/lukas-blecher" title="lukas-blecher"><img src="https://avatars.githubusercontent.com/u/55287601?v=4&s=64" width="32" height="32" alt="lukas-blecher"></a>
<a href="https://github.com/ian-adams" title="ian-adams"><img src="https://avatars.githubusercontent.com/u/43627295?v=4&s=64" width="32" height="32" alt="ian-adams"></a>
<a href="https://github.com/calclavia" title="calclavia"><img src="https://avatars.githubusercontent.com/u/1828968?v=4&s=64" width="32" height="32" alt="calclavia"></a>
<a href="https://github.com/mcree" title="mcree"><img src="https://avatars.githubusercontent.com/u/3463986?v=4&s=64" width="32" height="32" alt="mcree"></a>
<a href="https://github.com/AmirF194" title="AmirF194"><img src="https://avatars.githubusercontent.com/u/26088029?v=4&s=64" width="32" height="32" alt="AmirF194"></a>
<a href="https://github.com/LeptusHe" title="LeptusHe"><img src="https://avatars.githubusercontent.com/u/8146725?v=4&s=64" width="32" height="32" alt="LeptusHe"></a>
<a href="https://github.com/iamnotmili" title="iamnotmili"><img src="https://avatars.githubusercontent.com/u/136418159?v=4&s=64" width="32" height="32" alt="iamnotmili"></a>
<a href="https://github.com/andrewroxby" title="andrewroxby"><img src="https://avatars.githubusercontent.com/u/130508986?v=4&s=64" width="32" height="32" alt="andrewroxby"></a>
<a href="https://github.com/w-clary" title="w-clary"><img src="https://avatars.githubusercontent.com/u/19863024?v=4&s=64" width="32" height="32" alt="w-clary"></a>
<a href="https://github.com/JohanVisser97" title="JohanVisser97"><img src="https://avatars.githubusercontent.com/u/190853871?v=4&s=64" width="32" height="32" alt="JohanVisser97"></a>
<a href="https://github.com/feima3333" title="feima3333"><img src="https://avatars.githubusercontent.com/u/103805006?v=4&s=64" width="32" height="32" alt="feima3333"></a>
<a href="https://github.com/ElliotRoe" title="ElliotRoe"><img src="https://avatars.githubusercontent.com/u/32718462?v=4&s=64" width="32" height="32" alt="ElliotRoe"></a>
<a href="https://github.com/dshushin" title="dshushin"><img src="https://avatars.githubusercontent.com/u/59300536?v=4&s=64" width="32" height="32" alt="dshushin"></a>
<a href="https://github.com/cywwycedward" title="cywwycedward"><img src="https://avatars.githubusercontent.com/u/62976285?v=4&s=64" width="32" height="32" alt="cywwycedward"></a>
<a href="https://github.com/AndreiPashkin" title="AndreiPashkin"><img src="https://avatars.githubusercontent.com/u/4378647?v=4&s=64" width="32" height="32" alt="AndreiPashkin"></a>
<a href="https://github.com/6801318d8d" title="6801318d8d"><img src="https://avatars.githubusercontent.com/u/144167388?v=4&s=64" width="32" height="32" alt="6801318d8d"></a>
<a href="https://github.com/bunop" title="bunop"><img src="https://avatars.githubusercontent.com/u/5947792?v=4&s=64" width="32" height="32" alt="bunop"></a>
<a href="https://github.com/riichard" title="riichard"><img src="https://avatars.githubusercontent.com/u/616976?v=4&s=64" width="32" height="32" alt="riichard"></a>
<a href="https://github.com/SipengXie2024" title="SipengXie2024"><img src="https://avatars.githubusercontent.com/u/184713193?v=4&s=64" width="32" height="32" alt="SipengXie2024"></a>
<a href="https://github.com/brushax" title="brushax"><img src="https://avatars.githubusercontent.com/u/56171752?v=4&s=64" width="32" height="32" alt="brushax"></a>
<a href="https://github.com/TomBener" title="TomBener"><img src="https://avatars.githubusercontent.com/u/49151155?v=4&s=64" width="32" height="32" alt="TomBener"></a>
<a href="https://github.com/braininahat" title="braininahat"><img src="https://avatars.githubusercontent.com/u/15668020?v=4&s=64" width="32" height="32" alt="braininahat"></a>
<a href="https://github.com/linxule" title="linxule"><img src="https://avatars.githubusercontent.com/u/43122877?v=4&s=64" width="32" height="32" alt="linxule"></a>
<a href="https://github.com/yuanjua" title="yuanjua"><img src="https://avatars.githubusercontent.com/u/80858000?v=4&s=64" width="32" height="32" alt="yuanjua"></a>
<a href="https://github.com/h4rvey-g" title="h4rvey-g"><img src="https://avatars.githubusercontent.com/u/32609824?v=4&s=64" width="32" height="32" alt="h4rvey-g"></a>
<a href="https://github.com/aronnaxlin" title="aronnaxlin"><img src="https://avatars.githubusercontent.com/u/87559395?v=4&s=64" width="32" height="32" alt="aronnaxlin"></a>
<a href="https://github.com/aur3l14no" title="aur3l14no"><img src="https://avatars.githubusercontent.com/u/12196273?v=4&s=64" width="32" height="32" alt="aur3l14no"></a>
<a href="https://github.com/bkmzhmtd" title="bkmzhmtd"><img src="https://avatars.githubusercontent.com/u/31477377?v=4&s=64" width="32" height="32" alt="bkmzhmtd"></a>
<a href="https://github.com/feiiiiii5" title="feiiiiii5"><img src="https://avatars.githubusercontent.com/u/204683769?v=4&s=64" width="32" height="32" alt="feiiiiii5"></a>
<a href="https://github.com/jianxing-chen" title="jianxing-chen"><img src="https://avatars.githubusercontent.com/u/24669718?v=4&s=64" width="32" height="32" alt="jianxing-chen"></a>
<a href="https://github.com/jisoopark-tamu" title="jisoopark-tamu"><img src="https://avatars.githubusercontent.com/u/242275937?v=4&s=64" width="32" height="32" alt="jisoopark-tamu"></a>
<a href="https://github.com/minhna1112" title="minhna1112"><img src="https://avatars.githubusercontent.com/u/26354139?v=4&s=64" width="32" height="32" alt="minhna1112"></a>
<a href="https://github.com/patrickjcrawford" title="patrickjcrawford"><img src="https://avatars.githubusercontent.com/u/89989667?v=4&s=64" width="32" height="32" alt="patrickjcrawford"></a>
<a href="https://github.com/RomanRietsche" title="RomanRietsche"><img src="https://avatars.githubusercontent.com/u/22812524?v=4&s=64" width="32" height="32" alt="RomanRietsche"></a>
<a href="https://github.com/skhyun-ocean" title="skhyun-ocean"><img src="https://avatars.githubusercontent.com/u/263582378?v=4&s=64" width="32" height="32" alt="skhyun-ocean"></a>
<a href="https://github.com/whliao5am" title="whliao5am"><img src="https://avatars.githubusercontent.com/u/44702685?v=4&s=64" width="32" height="32" alt="whliao5am"></a>
<a href="https://github.com/AndyNieubourg" title="AndyNieubourg"><img src="https://avatars.githubusercontent.com/u/6673446?v=4&s=64" width="32" height="32" alt="AndyNieubourg"></a>
<a href="https://github.com/L3ulll" title="L3ulll"><img src="https://avatars.githubusercontent.com/u/270812873?v=4&s=64" width="32" height="32" alt="L3ulll"></a>
<a href="https://github.com/ChadThackray" title="ChadThackray"><img src="https://avatars.githubusercontent.com/u/67918202?v=4&s=64" width="32" height="32" alt="ChadThackray"></a>
<a href="https://github.com/chengzhag" title="chengzhag"><img src="https://avatars.githubusercontent.com/u/9084912?v=4&s=64" width="32" height="32" alt="chengzhag"></a>
<a href="https://github.com/claude" title="claude"><img src="https://avatars.githubusercontent.com/u/81847?v=4&s=64" width="32" height="32" alt="claude"></a>
<a href="https://github.com/rafaelcorsi" title="rafaelcorsi"><img src="https://avatars.githubusercontent.com/u/1039615?v=4&s=64" width="32" height="32" alt="rafaelcorsi"></a>
<a href="https://github.com/ebarkhordar" title="ebarkhordar"><img src="https://avatars.githubusercontent.com/u/22658149?v=4&s=64" width="32" height="32" alt="ebarkhordar"></a>
<a href="https://github.com/floriancaro" title="floriancaro"><img src="https://avatars.githubusercontent.com/u/34598596?v=4&s=64" width="32" height="32" alt="floriancaro"></a>
<a href="https://github.com/MarvinGalway" title="MarvinGalway"><img src="https://avatars.githubusercontent.com/u/258646604?v=4&s=64" width="32" height="32" alt="MarvinGalway"></a>
<a href="https://github.com/supersistence" title="supersistence"><img src="https://avatars.githubusercontent.com/u/33341498?v=4&s=64" width="32" height="32" alt="supersistence"></a>
<a href="https://github.com/igorcosta" title="igorcosta"><img src="https://avatars.githubusercontent.com/u/1169752?v=4&s=64" width="32" height="32" alt="igorcosta"></a>
<a href="https://github.com/jacobtfisher" title="jacobtfisher"><img src="https://avatars.githubusercontent.com/u/21992882?v=4&s=64" width="32" height="32" alt="jacobtfisher"></a>
<a href="https://github.com/jaehho" title="jaehho"><img src="https://avatars.githubusercontent.com/u/99066809?v=4&s=64" width="32" height="32" alt="jaehho"></a>
<a href="https://github.com/28Smiles" title="28Smiles"><img src="https://avatars.githubusercontent.com/u/9430219?v=4&s=64" width="32" height="32" alt="28Smiles"></a>
<a href="https://github.com/LETHEVIET" title="LETHEVIET"><img src="https://avatars.githubusercontent.com/u/50667900?v=4&s=64" width="32" height="32" alt="LETHEVIET"></a>
<a href="https://github.com/menyoung" title="menyoung"><img src="https://avatars.githubusercontent.com/u/5209088?v=4&s=64" width="32" height="32" alt="menyoung"></a>
<a href="https://github.com/schmidma" title="schmidma"><img src="https://avatars.githubusercontent.com/u/7946935?v=4&s=64" width="32" height="32" alt="schmidma"></a>
<a href="https://github.com/muhammedhunaid" title="muhammedhunaid"><img src="https://avatars.githubusercontent.com/u/96192659?v=4&s=64" width="32" height="32" alt="muhammedhunaid"></a>
<a href="https://github.com/nielsarts" title="nielsarts"><img src="https://avatars.githubusercontent.com/u/31060548?v=4&s=64" width="32" height="32" alt="nielsarts"></a>
<a href="https://github.com/strelkon" title="strelkon"><img src="https://avatars.githubusercontent.com/u/25686564?v=4&s=64" width="32" height="32" alt="strelkon"></a>
</p>
<!-- contributors:end -->

## 📄 License

MIT
