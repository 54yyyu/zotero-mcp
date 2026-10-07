// The settings half of the host, shared by the panel (host.ts builds on it) and Zotero's Settings pane (prefpane.ts).
// Settings live in one pref; each host caches it parsed, and a pref observer drops the cache and tells the UI when
// another window wrote it, so a change in the pane shows in every open panel at once, and the other way round.
import type { BackendId, PanelSettings, SettingsHost } from "../types.ts";
import { withDefaults } from "./defaults.ts";
import { createImages } from "./image.ts";
import { createSkills } from "./skills.ts";
import * as keychain from "./keychain.ts";
import { PREFIX, prefs } from "./settings.ts";
import { createStore } from "./store.ts";

export function createSettingsHost(opts: { version: string; win: any; dataDir: string }) {
  const { win, dataDir } = opts;
  // Where chats run. Visible and predictable (~/Documents/Zotero-Agent) rather than buried in the profile, so a chat can
  // be continued from a terminal; the user can point it anywhere in the settings.
  // ZMC_DEFAULT_CHAT_FOLDER is for the test harness, which must never write into the real home.
  const defaultFolder = (): string => {
    const forced = Services.env.get("ZMC_DEFAULT_CHAT_FOLDER");
    if (forced) return forced;
    const home = Services.dirsvc.get("Home", Ci.nsIFile);
    const docs = home.clone(); docs.append("Documents");
    return PathUtils.join(docs.exists() ? docs.path : home.path, "Zotero-Agent");
  };
  const chatFolder = () => settings().chatFolder || defaultFolder();
  const store = createStore(PathUtils.join(dataDir, "sessions"));
  const images = createImages(win, dataDir);
  const skills = createSkills({ dataDir, win, defaultFolder });

  // Read on every context change, so parsed once; dropped whenever the pref changes, from here or elsewhere.
  let cached: PanelSettings | null = null;
  const settings = (): PanelSettings => (cached ??= withDefaults(prefs.json<Partial<PanelSettings>>("settings", {}), Zotero.locale));
  const listeners = new Set<() => void>();
  let writing = false;
  const observer = Zotero.Prefs.registerObserver(`${PREFIX}settings`, () => {
    cached = null;
    if (!writing) for (const cb of [...listeners]) cb();
  }, true);
  const write = (value: string) => {
    writing = true; // the observer runs inside set(); our own save is not news to our own UI
    try { prefs.set("settings", value); } finally { writing = false; }
    cached = null;
  };
  const dark = win.matchMedia("(prefers-color-scheme: dark)");

  // Everything but the runtime, which the panel and the pane each bring.
  const host: Omit<SettingsHost, "runtime"> = {
    getSettings: settings,
    async setSettings(patch) { write(JSON.stringify({ ...prefs.json("settings", {}), ...patch })); },
    async resetSettings() { write(""); await images.remove(); },
    setApiKey: keychain.setApiKey,
    hasApiKey: async (b: BackendId) => !!(await keychain.getApiKey(b)),
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
    chooseImage: () => images.choose(),
    loadImage: () => images.load(),
    removeImage: () => images.remove(),
    skills: skills.host,
    theme: () => (dark.matches ? "dark" : "light"),
    onThemeChange(cb) {
      dark.addEventListener("change", cb);
      return () => dark.removeEventListener("change", cb);
    },
    onSettingsChange(cb) {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
  };
  return {
    host, settings, store, images, skills, chatFolder, defaultFolder,
    dispose() { Zotero.Prefs.unregisterObserver(observer); listeners.clear(); },
  };
}
