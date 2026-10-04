// PanelHost: what ui/ gets. This is the only place that knows both the UI's needs and Zotero's APIs.
import type { BackendId, ContextChip, PanelHost, PanelSettings, Spawner } from "../types.ts";
import { buildBrief, createRuntime, findBinary, prepareWorkspace, resumeCommand } from "../agent/index.ts";
import { ContextTracker } from "./context.ts";
import { withDefaults } from "./defaults.ts";
import { describeContext } from "./describe.ts";
import { dropChips } from "./drop.ts";
import { createDoctor } from "./doctor.ts";
import * as keychain from "./keychain.ts";
import { openTarget } from "./open.ts";
import { chipForHit, search } from "./search.ts";
import { createGeckoSpawner } from "./spawn-gecko.ts";
import { prefs } from "./settings.ts";
import { createStore } from "./store.ts";

export interface HostBundle {
  host: PanelHost;
  spawner: Spawner;
  context: ContextTracker;
  dispose(): void;
}

export function createHost(opts: { id: string; version: string; win: any; dataDir: string }): HostBundle {
  const { win, dataDir } = opts;
  // Where chats run. Visible and predictable (~/Documents/Zotero Chat) rather than buried in the profile, so a chat can
  // be continued from a terminal; the user can point it anywhere in the settings.
  // ZMC_DEFAULT_CHAT_FOLDER is for the test harness, which must never write into the real home.
  const defaultFolder = (): string => {
    const forced = Services.env.get("ZMC_DEFAULT_CHAT_FOLDER");
    if (forced) return forced;
    const home = Services.dirsvc.get("Home", Ci.nsIFile);
    const docs = home.clone(); docs.append("Documents");
    return PathUtils.join(docs.exists() ? docs.path : home.path, "Zotero Chat");
  };
  const chatFolder = () => settings().chatFolder || defaultFolder();
  const spawner = createGeckoSpawner();
  const context = new ContextTracker(opts.id);
  context.start(win);
  const runtime = createRuntime({ spawner, bridgeDir: PathUtils.join(dataDir, "bridges") });
  const store = createStore(PathUtils.join(dataDir, "sessions"));

  // Read on every context change, so parsed once and dropped when something saves.
  let cached: PanelSettings | null = null;
  const settings = (): PanelSettings => (cached ??= withDefaults(prefs.json<Partial<PanelSettings>>("settings", {})));
  const doctor = createDoctor({ win, spawner, runtime, settings });
  const dark = win.matchMedia("(prefers-color-scheme: dark)");

  const host: PanelHost = {
    runtime,
    currentContext() {
      const s = settings();
      if (!s.followFocus) return [];
      return context.current(win)
        .filter((c) => s.attachSelection || c.kind !== "selection")
        .map((c) => (c.kind === "area" && !s.attachAreas ? { ...c, image: undefined } : c));
    },
    onContextChange: (cb) => context.on(cb),
    search: (q) => search(win, q, context.activeReader(win)?._item ?? null),
    async chipFor(hit) {
      let text: string | undefined;
      if (hit.kind === "annotation" && hit.ref.annotationKey) {
        const a = Zotero.Items.getByLibraryAndKey(hit.ref.libraryID, hit.ref.annotationKey);
        text = a?.annotationText || a?.annotationComment || undefined;
      }
      return chipForHit(hit, text);
    },
    open: (target) => openTarget(win, target),
    dropChips: (data) => dropChips(context, data),
    describeContext: (chips: ContextChip[]) => describeContext(chips),

    getSettings: settings,
    async setSettings(patch) {
      prefs.setJson("settings", { ...prefs.json("settings", {}), ...patch });
      cached = null;
      context.refresh();
    },
    async resetSettings() {
      prefs.set("settings", "");
      cached = null;
      context.refresh();
    },
    setApiKey: keychain.setApiKey,
    hasApiKey: async (b: BackendId) => !!(await keychain.getApiKey(b)),

    sessions: () => store.sessions(),
    loadEvents: (id) => store.loadEvents(id),
    appendEvent: (s, ev) => store.appendEvent(s, ev),
    deleteSession: (id) => store.deleteSession(id),
    clearHistory: () => store.clearAll(),
    async revealWorkspace() {
      await IOUtils.makeDirectory(chatFolder(), { createAncestors: true, ignoreExisting: true });
      Zotero.File.reveal(chatFolder());
    },
    about: () => ({ version: opts.version, workspace: chatFolder() }),

    async chooseFolder(start) {
      const fp = Cc["@mozilla.org/filepicker;1"].createInstance(Ci.nsIFilePicker);
      fp.init(win.browsingContext, "Choose the chat folder", Ci.nsIFilePicker.modeGetFolder);
      if (start) try { fp.displayDirectory = Zotero.File.pathToFile(start); } catch { /* a folder that no longer exists */ }
      const result: number = await new Promise((r) => fp.open(r));
      return result === Ci.nsIFilePicker.returnOK ? fp.file.path : null;
    },
    resumeCommand: (s) => resumeCommand(s.backend, s.cwd, s.agentSessionId),

    doctor,
    async prepareSession(resumeIn) {
      const env = await spawner.baseEnv();
      const zoteroCli = await findBinary(spawner, env, "zotero-cli");
      // A chat being resumed runs in the folder it started in; agents look a session up by its folder.
      const cwd = await prepareWorkspace(spawner, resumeIn || chatFolder(), zoteroCli ? { zoteroCli } : {});
      const s = settings();
      const key = s.auth[s.backend] === "api-key" ? await keychain.getApiKey(s.backend) : null;
      const varName = keychain.API_KEY_ENV[s.backend];
      return { cwd, brief: buildBrief(), env: key && varName ? { [varName]: key } : {} };
    },

    theme: () => (dark.matches ? "dark" : "light"),
    onThemeChange(cb) {
      dark.addEventListener("change", cb);
      return () => dark.removeEventListener("change", cb);
    },
  };
  return { host, spawner, context, dispose: () => context.stop() };
}
