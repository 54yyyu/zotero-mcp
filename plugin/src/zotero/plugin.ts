// Entry point: bootstrap.js loads the bundle and calls createPlugin().
import { injectPanel, type Injected } from "./inject.ts";
import type { Panel } from "./panel.ts";
import { prefs } from "./settings.ts";
import { maybeRunTestScript } from "./testrunner.ts";

interface InitArgs { id: string; version: string; rootURI: string }

/** prefpane.js (run by Zotero's Settings window when our pane opens) hands us the pane's root element on this topic. */
const PANE_TOPIC = "zotero-chat:prefpane";

class ChatPlugin {
  readonly windows = new Map<any, Injected>();
  readonly init: InitArgs;
  /** Milliseconds spent reading plugin.js, in startup (of it, registering the Settings pane), and in loading panel.js for the panel or the pane: the budget tests' numbers. */
  readonly timing = { loadMs: 0, startupMs: 0, paneRegisterMs: 0, panelLoadMs: null as number | null, paneLoadMs: null as number | null };
  private panelModule: any = null;

  constructor(init: InitArgs) {
    this.init = init;
  }

  async startup(): Promise<void> {
    const t0 = Date.now();
    for (const win of Zotero.getMainWindows()) this.onMainWindowLoad(win);
    const t1 = Components.utils.now();
    this.registerPane();
    this.timing.paneRegisterMs = Components.utils.now() - t1;
    Zotero.Reader.registerEventListener("renderTextSelectionPopup", this.onSelectionPopup, this.init.id); // Zotero drops it at shutdown
    this.timing.startupMs = Date.now() - t0;
    const first = Zotero.getMainWindow();
    // The harness hook (loads the script named by ZMC_TEST_SCRIPT) exists in dev builds only: the released plugin has no such door.
    if (__TEST_HOOKS__ && first) await maybeRunTestScript(this, first);
  }

  /** panel.js is a second bundle, read on first use so that Zotero's startup pays only for the container and button. */
  private loadPanel(win: any, shadow: ShadowRoot): Panel {
    if (!this.panelModule) {
      const t0 = Date.now();
      const scope: any = { Zotero, Services, ChromeUtils, IOUtils, PathUtils, Components, Cc, Ci };
      Services.scriptloader.loadSubScript(this.init.rootURI + "panel.js", scope);
      this.panelModule = scope.ZoteroChatPanel;
      this.timing.panelLoadMs = Date.now() - t0;
    }
    return this.panelModule.createPanel({ id: this.init.id, version: this.init.version, win, dataDir: PathUtils.join(Zotero.Profile.dir, "zotero-chat"), shadow });
  }

  /**
   * A "Zotero Chat" pane in Zotero's Settings. Registering adds an entry to Zotero's list (not awaited: Zotero resolves the
   * icon later); the pane's markup, its script and panel.js are read only when the user opens the pane.
   */
  private registerPane(): void {
    Services.obs.addObserver(this.paneObserver, PANE_TOPIC);
    Zotero.PreferencePanes.register({ pluginID: this.init.id, id: "zotero-chat-pane", label: "Zotero Chat", src: "prefpane.xhtml", scripts: ["prefpane.js"] })
      .catch((e: unknown) => Zotero.logError(e));
  }

  private paneObserver = { observe: (subject: any) => { const { root, win } = subject.wrappedJSObject; this.mountPane(win, root); } };

  /**
   * The pane gets a copy of panel.js of its own: the UI keeps the window and document it draws in as module state, and
   * the pane's are the Settings window's, not the main window's. It shows a catalog only if an open panel has read it.
   */
  private mountPane(win: any, root: any): void {
    const t0 = Date.now();
    const scope: any = { Zotero, Services, ChromeUtils, IOUtils, PathUtils, Components, Cc, Ci };
    Services.scriptloader.loadSubScript(this.init.rootURI + "panel.js", scope);
    const known = (b: string) => { for (const w of this.windows.values()) { const c = w.loaded()?.bundle.knownCatalog(b as never); if (c) return c; } return undefined; };
    const pane = scope.ZoteroChatPanel.createSettingsPane({ version: this.init.version, win, dataDir: PathUtils.join(Zotero.Profile.dir, "zotero-chat"), root, known });
    this.timing.paneLoadMs = Date.now() - t0;
    // Zotero sends "unload" to the pane's elements when the Settings window closes.
    root.addEventListener("unload", () => pane.dispose(), { once: true });
  }

  /** "Ask in chat" in the reader's text selection popup, styled like its own buttons. Nothing loads until it is pressed. */
  private onSelectionPopup = (event: any): void => {
    if (!event.params?.annotation?.text?.trim()) return;
    const b = event.doc.createElement("button");
    b.className = "toolbar-button wide-button";
    b.setAttribute("data-tabstop", "1");
    b.textContent = "Ask in chat";
    b.addEventListener("click", () => {
      const injected = this.windows.get(Zotero.getMainWindow());
      if (!injected) return;
      injected.show();
      injected.ensure().askAbout(event);
    });
    event.append(b);
  };

  /** The loaded panel of a window (loading it if need be). */
  panel(win: any = Zotero.getMainWindow()): Panel {
    return this.windows.get(win)!.ensure();
  }

  onMainWindowLoad(win: any): void {
    if (this.windows.has(win)) return;
    const injected = injectPanel(win, (shadow) => this.loadPanel(win, shadow), prefs);
    if (injected) this.windows.set(win, injected);
  }

  onMainWindowUnload(win: any): void {
    this.windows.get(win)?.remove();
    this.windows.delete(win);
  }

  async shutdown(): Promise<void> {
    Services.obs.removeObserver(this.paneObserver, PANE_TOPIC); // Zotero drops the pane itself when a plugin shuts down
    for (const win of [...this.windows.keys()]) this.onMainWindowUnload(win);
  }
}

export function createPlugin(init: InitArgs): ChatPlugin {
  return new ChatPlugin(init);
}
