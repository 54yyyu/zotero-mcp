# Zotero chat plugin

A chat panel inside Zotero, on the right of the library and of every reader tab. It talks to your own AI agent (Claude Code, Codex or pi), tells it what you have open (the selected item, the page you are on, the text or area you selected), and lets it work your library through [`zotero-cli`](cli.md) and its skill. Answers cite papers with real `zotero://` links, so a click opens the PDF at that page.

The agent is not restricted by the panel: it can search the web, run shell commands and read files, as it would in a terminal. Permission requests show up as cards in the chat, and a mode setting chooses between asking every time and allowing everything.

## Install

You need Zotero 7 or newer (developed and tested on Zotero 10), [`zotero-cli`](cli.md) on your PATH (`uv tool install zotero-mcp-server`), and Node.js. The plugin ships inside the `zotero-mcp-server` wheel:

```bash
zotero-cli plugin           # where the .xpi is, and how to install it
zotero-cli plugin --path    # just the path, for scripts
```

In Zotero: **Tools > Plugins**, click the gear, **Install Plugin From File**, and choose that file. The plugin updates itself from `plugin/updates.json` in this repository. You can also download `zotero-chat.xpi` from the [GitHub release](https://github.com/54yyyu/zotero-mcp/releases/latest).

From a source checkout there is no packaged copy; build it with `npm ci && npm run build` in `plugin/`, which writes `plugin/dist/zotero-chat.xpi`. `zotero-cli plugin` finds that one too.

## First run

Open the panel from the toolbar button. A first-run check lists what is missing and, where it can, offers a one-click fix:

- Zotero's local API is reachable. In Zotero's settings, turn on "Allow other applications on this computer to communicate with Zotero".
- Writes are authorized (Zotero 10 or newer). Run `zotero-mcp authorize-local` once and choose "Always Allow".
- `zotero-cli` is installed.
- Node.js is installed.
- An agent is installed and signed in.

The panel installs the agent's ACP bridge once into its own data folder with `npm install`, so the first run needs network access. After that, chats start without it.

## Agents and sign-in

| Agent | Bridge |
|---|---|
| Claude Code | `@agentclientprotocol/claude-agent-acp` |
| Codex | `@agentclientprotocol/codex-acp` |
| pi | `pi-acp` |

Claude Code and Codex each run in one of two ways, chosen per agent in the settings (pi has no subscription: it runs on the provider keys you have configured for it):

- **Subscription.** The agent uses the login it already has on this machine (sign in once in a terminal with `claude`, `codex` or `pi`). The panel shows the account the bridge reports, for example "Claude Max". No key is involved.
- **API key.** You paste a key; the plugin stores it in Zotero's login manager, never in preferences, and hands it to the bridge as an environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...).

The plugin never runs a model itself.

## Settings

Open them from the gear in the panel. Everything is per agent where it can differ between agents.

- **Agent.** Which agent new chats use, and for each one its default model, reasoning effort (Claude's effort, Codex's low to max, pi's thinking level) and permission mode. The lists come from the agent itself, so they show what your account really offers. The model and effort can also be changed per chat from the composer.
- **Sign-in.** Subscription or API key, per agent (see below).
- **Context.** Whether the panel follows what you are reading, whether selected text is attached automatically, and whether a selected area is attached as an image. An image is sent to the agent's provider and costs tokens.
- **Chat.** Whether Enter sends (otherwise Cmd/Ctrl+Enter does), whether the agent's thinking is shown, and whether tool steps start expanded.
- **Startup.** Whether the panel is open when Zotero starts (off by default, so Zotero starts exactly as before).
- **Custom prompts.** Up to four get a shortcut (Cmd+Ctrl+1 to 4 on a Mac, Ctrl+Alt+1 to 4 elsewhere).
- **Chat folder.** Where new chats run, with Choose, Use default and Open. See below.
- **Data.** Clear all history, and reset settings (which keeps your API keys and chats).

The panel opens and closes from the toolbar button, or with Cmd+Option+L (Ctrl+Alt+L elsewhere). You can drag items from the library list, or annotations from the reader's sidebar, onto the composer to attach them.

## The chat folder, and continuing a chat in a terminal

Every chat is a normal agent session that runs in a folder, by default `~/Documents/Zotero Chat`. The panel installs the `zotero-cli` skill there (under `.claude/skills` and `.agents/skills`, plus a marked block in `AGENTS.md`; it only changes what it marked, so if you pick a folder you already use, the rest of your files are left alone). Change the folder in the settings; it applies to new chats. Each chat remembers the folder it started in, so changing the setting never breaks an old one.

Because the agent keeps the session itself, you can continue a chat from a terminal. In the history list, **Copy terminal command** gives you the right command for that chat, for example:

```bash
cd '/Users/you/Documents/Zotero Chat' && claude --resume <session id>
```

Codex uses `codex resume <session id>` and pi uses `pi --session <session id>`. Two things to know: do not run the same chat in the panel and in a terminal at the same time, and what you add in the terminal does not show up in the panel's history list (the panel keeps its own copy of the conversation; the agent's context has it).

## Privacy

The plugin has no server and sends nothing itself. What leaves your machine is whatever the agent you chose sends to its own provider: your messages, the context block the panel adds to them (the open item, page and selection), and anything the agent reads or fetches while working, such as PDF text. The panel's chat history is stored locally in your Zotero profile, under `zotero-chat/sessions/`. The agent keeps its own record of each session as well (for Claude Code under `~/.claude/projects`, for Codex under `~/.codex/sessions`, for pi under `~/.pi/agent/sessions`); that is what makes terminal resume work, and it follows that agent's own settings and retention.

## Troubleshooting

- **The panel says Zotero's local API is not responding.** A Zotero that has been running for days can stop serving its local API (port 23119) while the window looks fine. Restart Zotero. If it is a fresh start, check the "Allow other applications" setting above.
- **The panel reports that writes are not authorized.** Run `zotero-mcp authorize-local` and choose "Always Allow". Zotero 9 and older cannot take local writes; see [Local library limitations](troubleshooting.md#local-library-limitations).
- **`zotero-cli` or `node` is reported missing although your terminal has it.** Zotero started from the Dock or Start menu has a short PATH. The plugin looks them up through your login shell; if yours does not put them on PATH (check `$SHELL -lc 'command -v node'`), fix that in your shell profile and restart Zotero.
- **A chat answers with nothing, or the panel says the agent finished without answering.** The model or its reasoning effort was probably refused (some servers reject a high reasoning level). Pick a lower effort or another model for that agent in the settings.
- **The agent is not signed in.** Run `claude`, `codex` or `pi` once in a terminal and log in, or switch that agent to API-key mode.
- **`zotero-cli plugin` says the xpi is not built.** You are running from a source checkout: build it as above, or download it from the release page.
