# Zotero chat plugin: design

A side panel inside Zotero (library view and reader tabs) that chats with an agent. The agent is a normal
ACP agent (Claude Code, Codex, pi) run on the user's own subscription, or on an API key. It uses `zotero-cli`
and its skill for everything Zotero. The panel adds three things: the agent's **context** (what is open, which
page, what is selected), a **good chat UI**, and **one-click setup**. Nothing in the agent is restricted: web,
paper search outside the library, shell, files: it does what it would do in a terminal.

Working name: `addon.json` holds the plugin's name and id; rename in one place.

## Layers (the contract is `src/types.ts`)

```
Zotero main window ── #tabs-deck lives in an hbox below the tab bar; we append  <splitter> + <vbox#zmc-panel>
                      to that hbox: the panel sits right of both the library and every reader tab, and
                      keeps its place when tabs switch (measured, spike 2).
 └ zotero/   Zotero glue: injection, toolbar button + shortcuts, PanelHost implementation, GeckoSpawner,
             context capture, storage, keychain, doctor.              (only layer that touches Zotero APIs)
             Two bundles: plugin.js (container, button, shortcuts; loaded at Zotero startup) and panel.js
             (host + agent runtime + UI; read the first time the panel opens). Closed panel = near-zero cost.
 └ ui/       Shadow-DOM chat UI + pure reducer. Gets a PanelHost.     (DOM only)
 └ agent/    ACP client, backends, bridge locator/installer, brief.   (no DOM, no Zotero; Spawner injected)
```

`agent/` and `ui/` never import each other; both import only `types.ts`. `zotero/` wires them.

### Facts measured on this machine (Zotero 10.0.5, Gecko 140, macOS)

- Plugins run `Subprocess.sys.mjs` fine (`ChromeUtils.importESModule("resource://gre/modules/Subprocess.sys.mjs")`).
- A Dock-launched app has a short PATH. Resolve `node`, `npx`, `claude`, `codex`, `pi`, `uv`, `zotero-cli` through the
  login shell (`$SHELL -lc`), then pass that PATH to every child.
- ACP bridge for Claude: npm `@agentclientprotocol/claude-agent-acp` (0.85.1 tested; `@zed-industries/claude-code-acp`
  is deprecated). Codex: `@agentclientprotocol/codex-acp`. pi: `pi-acp`. Verify with `npm view` before pinning.
- `initialize` (protocolVersion 1) then `session/new {cwd, mcpServers: []}` works, and the bridge reports the signed-in
  account (`_auth/status_update`: "Claude Max") so subscription mode needs no key. `session/new` returns `modes`
  (`default`, `acceptEdits`, `plan`, `auto`, `bypassPermissions`) and `configOptions`.
- `session/new` `_meta.systemPrompt = { append: "..." }` appends to the agent's system prompt. The bridge reads
  `settingSources: ["user","project","local"]`, so skills in the workspace's `.claude/skills` load.
- Zotero 10 refuses a manifest without `applications.zotero.update_url`.
- Never launch a second Zotero on the default data dir (it is the user's real library). See the harness rules below.

### More facts, measured while building (each one cost a debugging round)

- Zotero 10 removed `ZoteroPane.getSelectedCollection()`; use `getSelectedCollections()`.
- The reader's live page and selection state is `reader._internalReader._state.primaryViewStats`
  (`pageIndex`, `pageLabel`, `pagesCount`, `canCopy` = text is selected). There is no event for a page turn or a
  cleared selection, so the context tracker polls it every 400 ms, only while the panel has a listener.
  `renderTextSelectionPopup` carries the selected text. Synthetic DOM/pointer events cannot start a selection in Zotero's
  own PDF view, so the selection chip is tested by feeding the event; a real drag is a human check.
- A selected image annotation's PNG is `Zotero.Annotations.getCacheImagePath(ann)` (render it with
  `Zotero.PDFWorker.renderAttachmentAnnotations(attachmentID)` if absent).
- `Zotero.Reader.open(attachmentID, { pageLabel })` / `{ annotationID }` is how `zotero://open-pdf/...?page=N` links are followed.
- State read out of the reader (`_internalReader._state.*`) lives in the reader window's JS compartment. Iterate it or `Array.from` it before calling array methods: `flatMap` on such an array hands back the callback's arrays wrapped instead of flattened (a chip arrived as `{"0": chip}`).
- The PDF view's `navigate({ position: { pageIndex, rects } })` scrolls there and flashes the rects in the selection color for
  2 s (`_highlightPosition`), no annotation. A page's glyphs are `view._pdfPages[i].chars` after `await view._ensureBasicPageData(i)`
  (`c`, `rect`, `inlineRect`, `rotation`, `lineBreakAfter`, ...); `_lastView` can be the Reading Mode overlay, which has none.
- Zotero's localized `firstCreator` carries invisible bidi isolates ("⁨Bertrand⁩ and ⁨Mullainathan⁩"): strip them before they reach chips or prompts.
- A request from Zotero's own window to its own local server fails (NetworkError). The doctor asks from outside, with
  `curl`, which is also what zotero-cli is: it catches the server that has gone quiet.
- zotero-cli (pyzotero) only talks to port 23119; the doctor flags any other port.
- Subscription mode must strip `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` from the login env, or a key exported for other tools silently bills the API.
- `claude-agent-acp` 0.85.1: `session/set_model` is -32601; set the model with `session/set_config_option {configId:"model"}`.
  codex-acp speaks `set_model`. Codex and pi ignore `_meta.systemPrompt`, so the brief rides on the first prompt.

## ACP client (agent/)

Port the proven parts of `~/Documents/projects/meeting-buddy/src/agent/{jsonrpc,acp,backends}.ts`, drop what a panel does
not need (terminals, usage logs, hibernation, workers). Lessons carried over from there, each learned the hard way:

- Delete a parent Claude Code's session variables (`CLAUDECODE`, `CLAUDE_CODE_*`, `CLAUDE_PID`, `CLAUDE_EFFORT`) from the child env.
- Spawn the bridge so the whole process group can be killed; `close()` is SIGTERM, grace, SIGKILL, and waits.
- Bound the handshake; a request into a dead pipe rejects at once; a spawn failure rejects, never throws uncaught.
- The bridge's own default model may be old; send the chosen model explicitly after `session/new`/`session/load`.
- Tool names come from `_meta.claudeCode.toolName` joined on `toolCallId`, never from titles.
- The user's own `claude` binary may be newer than the one the bridge bundles: pass `CLAUDE_CODE_EXECUTABLE` when found
  (`~/.local/bin/claude`, or `command -v claude`).

Bridges are installed once into the plugin's data dir (`npm install --prefix <dir> <pkg>@<pinned>`), then run with the
user's `node`. No `npx` at chat time (slow, network). Missing `node` is a doctor failure with an explanation, not a crash.

API-key mode is just environment for the bridge (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...); we never run our own LLM loop.

Permission requests (`session/request_permission`) become `permission` events; the UI answers with
`respondPermission`. A mode setting maps to the bridge's modes (default asks, bypassPermissions allows everything).

## What the agent is told (agent/brief.ts, appended to the system prompt)

Short. It states: you are in a Zotero side panel; use `zotero-cli` (the skill is in the workspace) for the library;
the user's current focus arrives in each message as a `<zotero-context>` block; **cite with real Zotero links**
`[Pager et al. 2009, p.8](zotero://open-pdf/library/items/ATTKEY?page=8)` (groups: `zotero://open-pdf/groups/<id>/items/KEY?page=8`;
items without a PDF: `zotero://select/library/items/KEY`). The panel turns those links into citation chips and a click
opens the page. Because they are ordinary Zotero URIs they stay valid when pasted into a note. An optional `&quote=` (6 to 15 words
copied verbatim, URL-encoded) makes a click also flash that passage for 2 s: `zotero/quote.ts` finds it in the page's text
(letters and digits only, so spacing, hyphenation, ligatures and quote styles never decide; then its first or last 8 words;
then the pages either side) and the reader draws it; not found is a plain page jump. Zotero's own handler ignores the param.

The workspace (`<profile>/zotero-chat/workspace`) gets the skill via `zotero-mcp install-skill --target claude --target agents --root <workspace>`.

## UI (ui/)

Shadow DOM, vanilla TypeScript, no framework (small `h()` helper), `marked` for the Markdown lexer. **Never `innerHTML` with
agent text**: the panel runs in a privileged window and the agent reads the web. Build DOM from tokens; links may be
`http(s):` or `zotero:` only; images only `data:`; no raw HTML.

Features (Beaver's, measured from its demo video, rebuilt in our visual language, which is meeting-buddy's):

- Header: close, new chat, history, doctor/status dot, account label.
- Empty state: custom prompts with `Cmd+Ctrl+1..4` (mac) / `Ctrl+Alt+1..4`, editable.
- Composer: auto context chip for the current item (bookmark = pin), `Text Selection` chip when text is selected in the reader,
  `Selected Area` chip with thumbnail + Go to Annotation / Remove, `+` and `@` to attach items/collections/annotations,
  model picker, mode picker, Send / Stop (Esc), Enter sends, Shift+Enter newline, drop an annotation on it.
- Transcript: user bubble with chips; "Thinking" row; assistant Markdown streamed; tool steps as one collapsed line each
  (`Searched library · "…"`, expandable to input/output); permission cards (Allow once / Always / Deny); plan list; errors as
  notices with a fix button; citation chips for `zotero:` links; "N sources" fold listing every cited item once;
  copy / retry per answer; auto-scroll that stops when the user scrolls up (with a jump-to-bottom pill).
- Math: `$...$`, `$$...$$` rendered with KaTeX (lazy, M6).
- Light and dark; follows Zotero's theme (`host.theme()`); keyboard accessible; works from 300 px to 700 px wide.
- Unavailable states are designed too: no backend, not logged in, zotero-cli missing, Zotero's local API off (23119 silent: "restart Zotero").

## Zotero glue (zotero/)

- Injection: described above; toolbar button in `#zotero-tabs-toolbar`; `Cmd+Shift+L`-style toggle; remembers open/closed and width.
- Context: library selection (`ZoteroPane.getSelectedItems`), open reader (`Zotero.Reader._readers`, current page, selection via
  `registerEventListener("renderTextSelectionPopup")`), area capture (image annotations in the reader), annotation drag.
- Open: `ZoteroPane.loadURI("zotero://open-pdf/...")`.
- Storage: sessions under `<profile>/zotero-chat/sessions/` (jsonl of ChatEvents + index). Settings in prefs `extensions.zotero-chat.*`.
  API keys in the login manager (`Services.logins`), never in prefs.
- Doctor: Zotero local API reachable (a long-running Zotero stops serving on 23119; the fix is "restart Zotero"), write access
  authorized, `zotero-cli` present (fix: `uv tool install zotero-mcp-server`, falling back to `pipx`/`pip --user`), node present,
  a backend available and signed in.

## Test and dev harness (the rules that matter)

`plugin/scripts/dev.mjs` runs a throwaway Zotero. **Hard rules** (a spike once opened the real library; see memory):

Flags: `--script <file>` runs a test script, `--build` rebuilds, `--mock-agent` swaps the bridges for `test/mock-agent.mjs`
(no tokens, deterministic), `--keep` leaves the window up, `--debug` keeps Zotero's own log, `--keep-data` reuses the last
library (every run otherwise starts from an empty one). zotero-cli cannot be pointed at the harness (it only talks to 23119, the
user's real port, which the harness never binds), so agent + CLI end to end is checked on the real library, read-only.

1. The profile lives under `plugin/.dev/`; its `user.js` pins `extensions.zotero.useDataDir=true` and `extensions.zotero.dataDir=<plugin/.dev/data>`.
2. The script refuses to start if that path is inside `~/Zotero`, and after launch verifies with `lsof` that the process holds nothing under `~/Zotero`; it kills the instance and exits non-zero otherwise.
3. Local server port is set explicitly (`extensions.zotero.httpServer.port`), checked free first; never 23119.
4. It never kills a process it did not start. It never runs while pointed at the default data dir.
5. Test data is generated (a tiny PDF made with PyMuPDF), imported into the throwaway library only.

Layers of test: unit (`node --test`, runtime against `test/mock-agent.mjs`, a deterministic ACP agent); UI (Playwright against
`preview/`, with `ui/fake-host.ts`, light+dark, 320/420/700 px, screenshots looked at, not just taken); in-Zotero integration (harness
loads the plugin, runs a script, writes a JSON result, takes a window snapshot with `drawSnapshot`, quits); and one opt-in live
test with real Claude Code (`ZMC_LIVE=1`), one tiny prompt, no library writes.

## Budgets (the codebase stays fast, accurate and free of redundancy)

What matters is that the plugin never slows Zotero's own startup, so that is the number under test: with the panel closed,
Zotero waits for this plugin only while it reads `plugin.js` (6 KB) and runs `startup()`, measured at 0 to 1 ms and capped at 50
(`test/zotero/budget.js`). `panel.js` (UI, agent runtime, katex, marked: about 490 KB minified) is read the first time the panel opens
(about 35 ms, capped at 150) and never at startup. Bundle size is not otherwise a concern, within reason (hundreds of KB are fine,
tens of MB are not): `test/budget.test.ts` only catches a blow-up. Redundancy is the thing to avoid: no helper, setting or layer without a
user or a failure it protects against. Every agent-written layer gets a simplify pass before it is called done.

## Milestones

| | scope |
|---|---|
| M0 | spikes: subprocess + ACP handshake in Zotero, panel injection (done) |
| M1 | panel shell, Claude Code chat with current-item context |
| M2 | reader awareness (page, selection, area), citation chips that open the page |
| M3 | permission cards, mode setting, receipts for what the agent changed |
| M4 | custom prompts + shortcuts, `@` mentions, history (session/load) |
| M5 | Codex and pi backends, API-key mode, first-run doctor with one-click `zotero-cli` install |
| M6 | polish: KaTeX, themes, empty/error states, packaging (xpi in the wheel, `zotero-cli plugin`), docs |
