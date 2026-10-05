// The heavy half, loaded the first time the panel opens (panel.js): host, agent runtime and UI.
// plugin.js, which Zotero loads at startup, only owns the container, the button and the shortcuts.
import type { BackendId, Catalog, PanelHost } from "../types.ts";
import { createRuntime } from "../agent/index.ts";
import { withDefaults } from "./defaults.ts";
import { createHost, type HostBundle } from "./host.ts";
import * as keychain from "./keychain.ts";
import { PREFIX, prefs } from "./settings.ts";
import { createTranslator, type Translator } from "./translate.ts";
import { createPopupTranslate } from "./translate-view.ts";
import { createSettingsHost } from "./settings-host.ts";
import { createGeckoSpawner } from "./spawn-gecko.ts";
import { mountPanel } from "../ui/index.ts";
import { mountSettings } from "../ui/pane.ts";

export interface Panel {
  bundle: HostBundle;
  host: PanelHost;
  api: ReturnType<typeof mountPanel>;
  dispose(): void;
}

export function createPanel(opts: { id: string; version: string; win: any; dataDir: string; shadow: ShadowRoot }): Panel {
  const bundle = createHost(opts);
  const api = mountPanel(opts.shadow, bundle.host);
  return { bundle, host: bundle.host, api, dispose() { api.dispose(); bundle.dispose(); } };
}

/**
 * Zotero's Settings pane: the settings screen in `root` (an element of the Settings window), on its own settings host.
 * Nothing is started by opening it: detect() only looks for the CLIs, and a catalog is shown at once only when an open
 * panel has already read it (`known`); otherwise the Agent card offers a button.
 */
export function createSettingsPane(opts: { version: string; win: any; dataDir: string; root: HTMLElement; known(b: BackendId): Catalog | undefined }): { dispose(): void } {
  const sh = createSettingsHost(opts);
  const own = createRuntime({ spawner: createGeckoSpawner(), bridgeDir: PathUtils.join(opts.dataDir, "bridges") });
  const runtime = { detect: () => own.detect(), catalog: (b: BackendId) => { const k = opts.known(b); return k ? Promise.resolve(k) : own.catalog(b); } };
  const ui = mountSettings(opts.root.shadowRoot ?? opts.root.attachShadow({ mode: "open" }), { ...sh.host, runtime }, (b) => !!opts.known(b));
  return { dispose() { ui.dispose(); sh.dispose(); } };
}

/**
 * The reader's Translate, made on the first press: a translator with its own runtime and folder (`<dataDir>/translate`,
 * never the chat folder), and the result in the selection popup. The settings are read from the pref on each press; a
 * change there closes a session started for other ones.
 */
export function createTranslate(opts: { dataDir: string }): { toggle(event: any, box: Element, button: HTMLElement): void; revive(event: any, box: Element, button: HTMLElement): void; translator: Translator; dispose(): void } {
  const settings = () => withDefaults(prefs.json("settings", {}), Zotero.locale);
  const dir = PathUtils.join(opts.dataDir, "translate");
  const translator = createTranslator({
    runtime: createRuntime({ spawner: createGeckoSpawner(), bridgeDir: PathUtils.join(opts.dataDir, "bridges") }),
    settings,
    async cwd() { await IOUtils.makeDirectory(dir, { createAncestors: true, ignoreExisting: true }); return dir; },
    async env(s) {
      const name = keychain.API_KEY_ENV[s.backend];
      const key = s.auth[s.backend] === "api-key" && name ? await keychain.getApiKey(s.backend) : null;
      return key ? { [name!]: key } : {};
    },
  });
  const observer = Zotero.Prefs.registerObserver(`${PREFIX}settings`, () => translator.settingsChanged(), true);
  const view = createPopupTranslate(translator, { settings, copy: (text) => Zotero.Utilities.Internal.copyTextToClipboard(text) });
  return { toggle: view.toggle, revive: view.revive, translator, dispose() { Zotero.Prefs.unregisterObserver(observer); void translator.dispose(); } };
}
