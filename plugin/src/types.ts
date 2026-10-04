// The contract between the three layers of the chat plugin. Read plugin/DESIGN.md first.
//
//   agent/  speaks ACP to a bridge process and emits ChatEvents      (no DOM, no Zotero)
//   ui/     renders ChatEvents and collects input                     (DOM only, no Zotero, no Subprocess)
//   zotero/ glues the two to Zotero: context, reader, storage, spawn  (the only layer that touches Zotero APIs)
//
// Only erasable TypeScript (no enum, no parameter properties, no namespaces): node runs these files
// directly in tests and esbuild bundles them for Gecko. Relative imports carry the `.ts` extension.

// ───────────────────────────── Zotero references ─────────────────────────────

/** Enough to find a thing in Zotero again. Page numbers are 0-based `pageIndex` plus the printed `pageLabel`. */
export interface ZoteroRef {
  /** 1 = My Library; a group library's id otherwise. */
  libraryID: number;
  itemKey?: string;
  /** The PDF/EPUB/snapshot attachment, when the thing lives in one. */
  attachmentKey?: string;
  pageIndex?: number;
  pageLabel?: string;
  annotationKey?: string;
  collectionKey?: string;
}

/** A small card of context attached to the next message. */
export interface ContextChip {
  id: string;
  kind: "item" | "reader" | "selection" | "area" | "annotation" | "collection" | "note";
  /** What the chip says: "Bell 2017", "Text Selection", "Selected Area · p.19". */
  label: string;
  /** Added by the panel because the user's focus moved there; false = the user added it. */
  auto: boolean;
  /** Survives focus changes (the bookmark toggle on the chip). Auto chips that are not pinned are replaced. */
  pinned: boolean;
  ref: ZoteroRef;
  /** Selection / annotation text, or the passage the chip stands for. */
  text?: string;
  /** Selected area as a PNG (base64, no data: prefix). */
  image?: { mime: "image/png"; data: string };
}

/** What a transcript keeps of a chip (no image bytes). */
export type ChipSummary = Pick<ContextChip, "id" | "kind" | "label" | "ref"> & { text?: string };

// ───────────────────────────── Transcript events ─────────────────────────────

export type ToolKind = "read" | "edit" | "delete" | "move" | "search" | "execute" | "think" | "fetch" | "switch_mode" | "other";
export type ToolStatus = "pending" | "running" | "done" | "failed";

export interface PermissionOption {
  id: string;
  name: string;
  /** allow_once | allow_always | reject_once | reject_always (ACP's own kinds). */
  kind: string;
}

export interface Usage {
  inputTokens?: number;
  outputTokens?: number;
  /** USD, when the backend can price it. */
  costUsd?: number;
}

/**
 * Everything a conversation is made of. `agent/` emits these while a turn runs, `ui/` folds them into
 * a transcript (ui/transcript.ts `applyEvent`, a pure function), and a session's saved history is the
 * list of them, replayed through the same function. Events that carry an `id` are upserts.
 */
export type ChatEvent =
  | { t: "user"; id: string; text: string; chips: ChipSummary[] }
  | { t: "turn_start"; turn: string }
  | { t: "thought"; turn: string; delta: string }
  | { t: "text"; turn: string; delta: string }
  | { t: "tool"; turn: string; id: string; title: string; kind: ToolKind; status: ToolStatus; input?: unknown; output?: string; /** The tool's own name when the bridge reports one (Claude: "Bash", "Read"); fall back to title/kind. */ name?: string }
  | { t: "plan"; turn: string; entries: { content: string; status: "pending" | "in_progress" | "completed" }[] }
  | { t: "permission"; turn: string; id: string; title: string; kind: ToolKind; input?: unknown; options: PermissionOption[]; resolved?: string; name?: string }
  | { t: "turn_end"; turn: string; stop: "end_turn" | "cancelled" | "max_tokens" | "refusal" | "error"; usage?: Usage }
  | { t: "notice"; level: "info" | "warn" | "error"; message: string; hint?: string };

// ───────────────────────────── agent/ ─────────────────────────────

export type BackendId = "claude-code" | "codex" | "pi";

export interface ModelOption { id: string; name: string; description?: string }
export interface ModeOption { id: string; name: string; description?: string }

/** Whether a backend can run here, as `detect()` measured it. `reason` is shown to the user when it cannot. */
export interface BackendStatus {
  id: BackendId;
  label: string;
  available: boolean;
  /** "node not found", "claude is not logged in", ... */
  reason?: string;
  /** How the user is signed in, when the bridge says ("Claude Max"); empty for API-key mode. */
  account?: string;
}

export interface StartOpts {
  backend: BackendId;
  /** The agent's working directory (the chat workspace). */
  cwd: string;
  /** Append to the agent's system prompt (the brief: where it is, how to cite). */
  brief: string;
  /** Continue a saved conversation (ACP session/load). */
  resumeSessionId?: string;
  model?: string;
  /** A permission mode id from the backend's own list (Claude: default | acceptEdits | plan | auto | bypassPermissions; Codex: read-only | workspace-write | agent | agent-full-access). */
  mode?: string;
  /** Reasoning effort / thinking level, from the backend's own list (`Catalog.efforts`). */
  effort?: string;
  /**
   * How the user pays. "subscription" strips the backend's API-key variables from the login-shell env, so a key the user
   * exported for other tools cannot silently turn a subscription chat into API billing. "api-key" leaves env alone.
   */
  auth?: "subscription" | "api-key";
  /** Folders put first on the agent's PATH: where zotero-cli lives when the login PATH lacks it (`~/.local/bin` right after the plugin installed it). */
  path?: string[];
  /** A throwaway session (reading a catalog): the backend is asked not to keep it. Claude honours that; Codex and pi cannot, so their probe leaves one small session file. */
  ephemeral?: boolean;
  /** Extra environment for the bridge, e.g. ANTHROPIC_API_KEY for API mode. */
  env?: Record<string, string>;
}

export interface PromptInput {
  text: string;
  images?: { mime: string; data: string }[];
}

/** One running conversation with an agent. */
export interface AgentSession {
  readonly sessionId: string;
  readonly backend: BackendId;
  readonly account?: string;
  readonly supportsImages: boolean;
  models(): ModelOption[];
  currentModel(): string | undefined;
  modes(): ModeOption[];
  currentMode(): string | undefined;
  /** Reasoning effort levels the backend offers (Claude's effort, pi's thinking level, Codex's low..max); empty when it has none. */
  efforts(): ModeOption[];
  currentEffort(): string | undefined;
  setModel(id: string): Promise<void>;
  setMode(id: string): Promise<void>;
  setEffort(id: string): Promise<void>;
  /** Events for this session; returns the unsubscribe. Replayed history from session/load is NOT emitted here. */
  on(listener: (ev: ChatEvent) => void): () => void;
  /** Runs one turn; resolves when it ends (its `turn_end` has been emitted). Rejects only if the bridge died. */
  prompt(input: PromptInput): Promise<void>;
  cancel(): Promise<void>;
  /** Answer a `permission` event. */
  respondPermission(id: string, optionId: string | null): void;
  close(): Promise<void>;
}

/** What a backend offers, readable before any chat exists (the settings page lists these). */
export interface Catalog {
  models: ModelOption[];
  modes: ModeOption[];
  efforts: ModeOption[];
  model?: string;
  mode?: string;
  effort?: string;
}

export interface AgentRuntime {
  detect(): Promise<BackendStatus[]>;
  start(opts: StartOpts): Promise<AgentSession>;
  /** The backend's models, modes and efforts. Cached after the first call (which starts a short-lived bridge). */
  catalog(backend: BackendId): Promise<Catalog>;
}

// ─────────────────── process spawning (injected into agent/) ───────────────────

export interface Proc {
  readonly pid: number;
  write(data: string): void;
  /** Called with each complete stdout line (no newline). */
  onStdoutLine(cb: (line: string) => void): void;
  onStderr(cb: (chunk: string) => void): void;
  /** Resolves with the exit code (null when killed by a signal). */
  readonly exited: Promise<number | null>;
  /** SIGTERM, then SIGKILL after a grace period; resolves when gone. */
  kill(): Promise<void>;
}

/** agent/ never imports node:child_process or Gecko's Subprocess; it is handed one of these. */
export interface Spawner {
  spawn(command: string, args: string[], opts: { cwd?: string; env: Record<string, string> }): Promise<Proc>;
  /** Run to completion; stdout and stderr captured. */
  run(command: string, args: string[], opts: { cwd?: string; env: Record<string, string>; timeoutMs?: number }): Promise<{ code: number | null; stdout: string; stderr: string }>;
  /** The environment a Dock-launched app lacks: the user's login-shell PATH and friends, merged over the process env. */
  baseEnv(): Promise<Record<string, string>>;
}

// ───────────────────────────── ui/ ─────────────────────────────

export interface ItemHit { ref: ZoteroRef; title: string; subtitle: string; kind: "item" | "collection" | "annotation" }

export interface PromptEntry {
  id: string;
  title: string;
  text: string;
  /** 1–4: bound to Cmd+Ctrl+N (mac) / Ctrl+Alt+N. */
  slot?: number;
}

export interface PanelSettings {
  backend: BackendId;
  /** Per backend; "" means the backend's own default. Ids are the backend's, so they are never shared between backends. */
  model: Record<BackendId, string>;
  mode: Record<BackendId, string>;
  effort: Record<BackendId, string>;
  /** Per backend: run on the user's subscription (bridge login) or on an API key kept in the OS keychain. */
  auth: Record<BackendId, "subscription" | "api-key">;
  prompts: PromptEntry[];

  // What the agent is told about the user's focus.
  /** The panel follows the reader / library selection and attaches it. Off: only what the user adds with + or @. */
  followFocus: boolean;
  /** The text selected in the reader is attached automatically. */
  attachSelection: boolean;
  /** A selected area is attached as an image (costs tokens, and sends the picture to the agent's provider). */
  attachAreas: boolean;

  // How the chat behaves.
  /** true: Enter sends, Shift+Enter is a newline. false: Cmd/Ctrl+Enter sends. */
  enterToSend: boolean;
  showThinking: boolean;
  /** Tool steps start expanded. */
  expandTools: boolean;
  /** The panel is open when Zotero starts. */
  openAtStart: boolean;
  /** Where new chats run: the agent's working directory, and where the zotero-cli skill is installed. "" = the default folder. */
  chatFolder: string;
}

export interface SavedSession {
  id: string;
  title: string;
  backend: BackendId;
  /** The folder the chat ran in. Agents find a session by its folder, so resuming (here or in a terminal) needs this one, not the current setting. */
  cwd: string;
  /** The bridge's session id, for session/load. */
  agentSessionId: string;
  updatedAt: number;
}

export interface DoctorCheck {
  id: "zotero-api" | "write-access" | "cli" | "node" | "backend";
  ok: boolean;
  label: string;
  detail?: string;
  /** A fix the panel can run itself, named for its button ("Install zotero-cli"). */
  fix?: { label: string; run: () => AsyncIterable<string> };
}

/** What `ui/` needs from the outside world. Implemented by zotero/host.ts; ui/ ships a fake for preview and tests. */
export interface PanelHost {
  runtime: AgentRuntime;

  /** The chips for wherever the user's focus is now (open reader, library selection, text selection). */
  currentContext(): ContextChip[];
  onContextChange(cb: () => void): () => void;
  /** `@` mentions: items, collections and annotations matching `query`. */
  search(query: string): Promise<ItemHit[]>;
  chipFor(hit: ItemHit): Promise<ContextChip>;
  /** Chips for what was dropped on the composer: items dragged from the library list, annotations dragged from the reader's sidebar. Empty when it is neither. */
  dropChips(data: DataTransfer): Promise<ContextChip[]>;
  /** Jump to a Zotero URI (`zotero://open-pdf/...`, `zotero://select/...`), or to a ref. */
  open(target: string | ZoteroRef): Promise<void>;
  /** Turn the chips into the text the agent reads (item keys, page, selection) plus images. */
  describeContext(chips: ContextChip[]): { text: string; images: { mime: string; data: string }[] };

  getSettings(): PanelSettings;
  setSettings(patch: Partial<PanelSettings>): Promise<void>;
  /** Keys live in the OS keychain, never in prefs. */
  setApiKey(backend: BackendId, key: string | null): Promise<void>;
  hasApiKey(backend: BackendId): Promise<boolean>;

  sessions(): Promise<SavedSession[]>;
  loadEvents(sessionId: string): Promise<ChatEvent[]>;
  appendEvent(session: SavedSession, ev: ChatEvent): Promise<void>;
  deleteSession(sessionId: string): Promise<void>;

  /** Delete every saved chat. */
  clearHistory(): Promise<void>;
  /** Show the chat folder in the file manager. */
  revealWorkspace(): Promise<void>;
  /** Back to the defaults (keeps API keys and history). */
  resetSettings(): Promise<void>;
  /** `workspace` is the folder new chats run in (the setting, or the default). */
  about(): { version: string; workspace: string };

  doctor(): Promise<DoctorCheck[]>;
  /** The brief and folder for an agent session (the host knows where the CLI and skill are). `cwd` resumes a saved chat in the folder it ran in; omitted, the chat folder setting is used. */
  prepareSession(cwd?: string): Promise<{ cwd: string; brief: string; env: Record<string, string> }>;
  /** A folder picker; null when cancelled. */
  chooseFolder(start?: string): Promise<string | null>;
  /** The shell command that continues a saved chat in a terminal, e.g. `cd "…" && claude --resume ID`; null when the backend has none. */
  resumeCommand(session: SavedSession): string | null;
  /** "light" | "dark", and a callback for when Zotero's theme changes. */
  theme(): "light" | "dark";
  onThemeChange(cb: () => void): () => void;
}

/** The only function the UI layer exports to the bootstrap: render into a shadow root, return a disposer. */
export type MountPanel = (root: ShadowRoot, host: PanelHost) => { dispose(): void; focusComposer(): void; runPrompt(slot: number): void };
