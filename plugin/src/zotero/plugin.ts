// Entry point: bootstrap.js loads the bundle and calls createPlugin().
import { injectPanel, type Injected } from "./inject.ts";
import type { Panel } from "./panel.ts";
import { prefs } from "./settings.ts";
import { maybeRunTestScript } from "./testrunner.ts";

interface InitArgs { id: string; version: string; rootURI: string }

class ChatPlugin {
  readonly windows = new Map<any, Injected>();
  readonly init: InitArgs;
  /** Milliseconds spent reading plugin.js, in startup, and in loading panel.js: the budget test's numbers. */
  readonly timing: { loadMs: number; startupMs: number; panelLoadMs: number | null } = { loadMs: 0, startupMs: 0, panelLoadMs: null };
  private panelModule: any = null;

  constructor(init: InitArgs) {
    this.init = init;
  }

  async startup(): Promise<void> {
    const t0 = Date.now();
    for (const win of Zotero.getMainWindows()) this.onMainWindowLoad(win);
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
    for (const win of [...this.windows.keys()]) this.onMainWindowUnload(win);
  }
}

export function createPlugin(init: InitArgs): ChatPlugin {
  return new ChatPlugin(init);
}
