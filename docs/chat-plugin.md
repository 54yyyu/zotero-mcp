# Zotero chat plugin

A chat panel inside Zotero, on the right of the library and of every reader tab. It talks to your own AI agent (Claude Code, Codex or pi), tells it what you have open (the selected item, the page you are on, the text or area you selected), and lets it work your library through [`zotero-cli`](cli.md) and its skill. Answers cite papers with real `zotero://` links, so a click opens the PDF at that page.

The agent is not restricted by the panel: it can search the web, run shell commands and read files, as it would in a terminal. Permission requests show up as cards in the chat, and a mode setting chooses between asking every time and allowing everything.

## Install

You need Zotero 7 or newer (developed and tested on Zotero 10), [`zotero-cli`](cli.md) on your PATH (`uv tool install zotero-mcp-server`), and Node.js. The plugin ships inside the `zotero-mcp-server` wheel:

```bash
zotero-cli plugin           # where the .xpi is, and how to install it
zotero-cli plugin --path    # just the path, for scripts
zotero-cli plugin --reveal  # also show the file in your file manager
```

In Zotero: **Tools > Plugins**, click the gear, **Install Plugin From File**, and choose that file. The plugin updates itself from `plugin/updates.json` in this repository. You can also download `zotero-chat.xpi` from the [GitHub release](https://github.com/54yyyu/zotero-mcp/releases/latest).

### Let your agent do it

Paste this to Claude Code, Codex, pi or any other agent that has a shell:

```text
Install the Zotero chat plugin for me. Make sure zotero-cli is installed (`uv tool install zotero-mcp-server`,
or `pipx install zotero-mcp-server`), then run `zotero-cli plugin --reveal`: it prints the path of zotero-chat.xpi
and shows it in my file manager. If it says the file is missing, download
https://github.com/54yyyu/zotero-mcp/releases/latest/download/zotero-chat.xpi instead. Zotero cannot install a
plugin from the command line, so finish by telling me the one step I do myself: in Zotero, Tools > Plugins, the
gear, Install Plugin From File, and choose that file.
```

The agent does everything except the last click: Zotero offers no supported way to install a plugin from outside (opening the file with Zotero offers to import it as a library, and a plugin dropped into the profile folder arrives switched off), so the file picker stays a human step.

From a source checkout there is no packaged copy; build it with `npm ci && npm run build` in `plugin/`, which writes `plugin/dist/zotero-chat.xpi`. `zotero-cli plugin` finds that one too.

## First run

Open the panel from the toolbar button. The first time, a short welcome has you choose your agent (Claude Code, Codex or pi, each marked ready or not), then runs a check that lists what is missing and, where it can, offers a one-click fix. "Skip setup" leaves it for later; the dot at the top of the panel always shows what is missing. The check covers:

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

Open them from the gear in the panel, or in Zotero's own Settings window under **Zotero Chat** (the same cards; a change in either shows in the other at once). They are a stack of cards, and everything is per agent where it can differ between agents.

- **Agent.** Which agent new chats use, how it signs in (subscription or API key, per agent; see below), and for each one its default model, reasoning effort (Claude's effort, Codex's low to max, pi's thinking level) and permission mode. The lists come from the agent itself, so they show what your account really offers. The button under the message box (the model name, with the effort level beside it) opens one menu to switch the agent (a chat that already has messages asks first, since another agent starts a new chat), pick a model, and set the effort on a slider. **Default** marks the model the agent itself uses when you have not chosen one (pi's own default, Claude's, Codex's); choosing it again clears your choice, here and in the settings, where it is the Default option. A short list shows the first four models and the rest under More models. A long one (pi lists every OpenRouter model) has a search field: just start typing (or press /), every word must match the name or id, Down moves into the results, Enter picks the first match, Esc clears the search and then closes. Your current model and the default come first, then the providers you set up yourself, then the big cloud lists, each under its provider's name.
- **Appearance.** Glass (frosted, translucent cards and menus; on by default, off is the flat look), the accent colour (Mono, black and white, by default; red and five more swatches, or any colour from the picker or as a hex code; text on it stays readable), a background (Plain by default, a soft Glow, five gradients, or a picture of your own, with its visibility and blur), text size, and density. Changes show as you make them. A picture is shrunk to 1600 px and kept in your Zotero profile as `zotero-chat/background.jpg`; it is never synced, and Remove deletes it.
- **Context.** Whether the panel follows what you are reading, whether selected text is attached automatically, and whether a selected area is attached as an image. An image is sent to the agent's provider and costs tokens.
- **Chat.** Whether Enter sends (otherwise Cmd/Ctrl+Enter does), whether the agent's thinking is shown, whether tool steps start expanded, and whether the panel is open when Zotero starts (off by default, so Zotero starts exactly as before).
- **Skills and prompts.** Your skills and saved prompts in one list: pin, turn off, edit, delete, add. See below.
- **Chat folder.** Where new chats run, with Choose, Use default and Open. See below.
- **Data.** Clear all history, and reset settings (which keeps your API keys and chats).
- **About.** The version and the keyboard shortcuts.

The panel opens and closes from the toolbar button, or with Cmd+Option+L (Ctrl+Alt+L elsewhere). You can drag items from the library list, or annotations from the reader's sidebar, onto the composer to attach them.

## Skills and prompts

A **prompt** is a message you send often ("Summarize this paper in five sentences"). A **skill** is a longer set of
instructions your agent follows step by step, for example how you like a paper annotated: a folder with a `SKILL.md` of
plain Markdown (a name, a one-line description, then the steps), see [skills/](skills/README.md) for an example.

- Type `/` at the start of the message box for a menu of your skills and prompts (type to filter, arrows, Enter or Tab).
  A skill goes in as `/name `: add a sentence if you like ("/annotate-paper only the figures"), then send. A prompt puts
  its text in the box to send or edit. Once a chat is under way, **Summarise now** is listed too where the agent offers it.
- Pin up to four (a skill or a prompt): they are buttons above the message box on a new chat, and run from anywhere in
  Zotero with Cmd+Ctrl+1 to 4 (Ctrl+Alt+1 to 4 elsewhere).
- Every agent gets a skill the same way: the panel copies it into the chat folder (`.agents/skills` and `.claude/skills`)
  and tells the agent in plain words to read that file and follow it, with what you have open attached as usual.
- Settings > **Skills and prompts** lists them all. **Add skill…** takes a skill's `SKILL.md` (its folder comes along) or
  any `.md` file (it becomes a new skill; you give it a name and a line). Before anything is copied you see the full
  text and every file that is copied or left out: scripts, programs, links and hidden files are left out, and scripts
  are copied only if you tick that you have read them. **Create with the agent** starts a chat in which your agent asks a
  few questions and writes the skill for you (or tells you it is better as a prompt).
- Your skills live in your Zotero profile, `zotero-chat/skills/<name>/SKILL.md` (**Reveal** opens the folder). They
  work in every chat folder. Copies the panel puts in a folder you chose are marked, and a skill of the same name that
  you put there yourself is never replaced.

Skills tell your agent what to do, and it acts with your permissions: only add skills you trust or wrote. The panel never
downloads one.

## While the agent works

One small indicator says what the agent is doing: Thinking (in the thinking row, which you can open to read along), Searching your library (or the web), Reading page 7, Working, Writing the answer, or Waiting for your OK when a permission card needs you. A grid of dots in your accent colour moves in a different pattern for each. The line changes at most every 0.4 s, so it never flickers, and it is gone when the turn ends. With your system's "reduce motion" setting the dots stand still.

## Diagrams

Ask the agent to draw something (a pipeline, a 2x2, a causal graph) and it answers with a figure in the chat. The agent writes plain SVG; the panel draws it in your theme's colours, so it follows light and dark mode and your accent colour. Hover a figure for its toolbar: show the SVG source, copy it as an image, or save it as PNG (on white, 1400 px wide) or SVG. Anything in a drawing that is not a shape or text, such as scripts, links or external images, is removed before it is shown. **Add to a note** saves the figure as a Zotero note on the item you are looking at, the drawing embedded as an image.

## Notes, math and formatting

Under each answer, **Save as note** turns it into a Zotero note on the paper you are reading (or the selected item; with nothing selected, a standalone note in the current collection). The note keeps headings, lists, tables, code, links (citation links still open the page), math (Zotero's note editor typesets it) and drawings as images; **Open** selects it. On the last answer, the bulb asks the agent to explain again, intuition first.

Answers and notes share a little formatting beyond Markdown: underline, strikethrough, sub- and superscript, and Zotero's text and highlight colours. The agent knows it, and `zotero-cli notes create/update` take the same text, so it can also write formatted notes directly.

Hover a display formula for **Copy TeX**. A formula too wide for the panel shrinks a little, then scrolls sideways. While an answer streams, half-written formulas and tables wait until they are complete.

Text you select in the reader goes into the composer as a **Text Selection** chip by itself (remove the chip if you do not want it sent); in the composer's **+** menu, **This page** attaches the page you are on as an image.

## Translate

Select a word, a sentence or a paragraph in a PDF and press **Translate** in Zotero's selection popup: the translation appears in the same popup, under the buttons, as it is written. **Copy** copies it; the language under it translates this selection into another language without changing your setting; Esc, a click elsewhere or a new selection closes it, as always. In the settings (the **Translate** card, also in Zotero's Settings pane) you can hide the button, choose the language (by default Zotero's own), and the model: **Fastest available** picks the model your agent itself describes as fast (Claude's Haiku, Codex's Luna; for pi, your chat model), or choose any of the agent's models.

It uses the agent you chat with, in a session of its own: it is told only to translate (and never to follow instructions found in the text), it gets no tools, none of your chats, and none of your agent's settings or skills; Codex runs read-only, and any request to use a tool is refused. The session starts when the selection popup appears (it costs no tokens until you press Translate) and stays ready for about five minutes, so with Claude a translation usually starts appearing within a second.

## How full the chat is

Once the agent reports it, a small ring beside the permission mode shows how much of the agent's context window this chat uses (amber from 70%, red from 85%). Hover it for the number; click it for details: tokens used, messages, the last turn's tokens (and cost, if "Show tokens and cost" is on), how often the agent summarised older parts, and what context your last message sent. From there, New chat starts fresh, and with Claude, Summarise now asks the agent to summarise the conversation so far and keep going.

## The chat folder, and continuing a chat in a terminal

Every chat is a normal agent session that runs in a folder, by default `~/Documents/Zotero-Chat`. The panel installs the `zotero-cli` skill there (under `.claude/skills` and `.agents/skills`, plus a marked block in `AGENTS.md`; it only changes what it marked, so if you pick a folder you already use, the rest of your files are left alone). Change the folder in the settings; it applies to new chats. Each chat remembers the folder it started in, so changing the setting never breaks an old one.

Because the agent keeps the session itself, you can continue a chat from a terminal. In the history list, **Copy terminal command** gives you the right command for that chat, for example:

```bash
cd '/Users/you/Documents/Zotero-Chat' && claude --resume <session id>
```

Codex uses `codex resume <session id>` and pi uses `pi --session <session id>`. Two things to know: do not run the same chat in the panel and in a terminal at the same time, and what you add in the terminal does not show up in the panel's history list (the panel keeps its own copy of the conversation; the agent's context has it).

## Privacy

The plugin has no server and sends nothing itself. What leaves your machine is whatever the agent you chose sends to its own provider: your messages, the context block the panel adds to them (the open item, page and selection), and anything the agent reads or fetches while working, such as PDF text. The panel's chat history is stored locally in your Zotero profile, under `zotero-chat/sessions/`. The agent keeps its own record of each session as well (for Claude Code under `~/.claude/projects`, for Codex under `~/.codex/sessions`, for pi under `~/.pi/agent/sessions`); that is what makes terminal resume work, and it follows that agent's own settings and retention. Translate sends the selected text (up to 6,000 characters) to the same provider; Claude and pi keep no record of those sessions, Codex keeps its usual one.

## Troubleshooting

- **The panel says Zotero's local API is not responding.** A Zotero that has been running for days can stop serving its local API (port 23119) while the window looks fine. Restart Zotero. If it is a fresh start, check the "Allow other applications" setting above.
- **The panel reports that writes are not authorized.** Run `zotero-mcp authorize-local` and choose "Always Allow". Zotero 9 and older cannot take local writes; see [Local library limitations](troubleshooting.md#local-library-limitations).
- **`zotero-cli` or `node` is reported missing although your terminal has it.** Zotero started from the Dock or Start menu has a short PATH. The plugin looks them up through your login shell; if yours does not put them on PATH (check `$SHELL -lc 'command -v node'`), fix that in your shell profile and restart Zotero.
- **A chat answers with nothing, or the panel says the agent finished without answering.** The model or its reasoning effort was probably refused (some servers reject a high reasoning level). Pick a lower effort or another model for that agent in the settings.
- **The agent is not signed in.** Run `claude`, `codex` or `pi` once in a terminal and log in, or switch that agent to API-key mode.
- **`zotero-cli plugin` says the xpi is not built.** You are running from a source checkout: build it as above, or download it from the release page.
