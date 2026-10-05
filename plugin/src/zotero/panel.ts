// The heavy half, loaded the first time the panel opens (panel.js): host, agent runtime and UI.
// plugin.js, which Zotero loads at startup, only owns the container, the button and the shortcuts.
import type { BackendId, Catalog, PanelHost } from "../types.ts";
import { createRuntime } from "../agent/index.ts";
import { createHost, type HostBundle } from "./host.ts";
import { createSettingsHost } from "./settings-host.ts";
import { createGeckoSpawner } from "./spawn-gecko.ts";
import { mountPanel } from "../ui/index.ts";
import { mountSettings } from "../ui/pane.ts";

export interface Panel {
  bundle: HostBundle;
  host: PanelHost;
  api: ReturnType<typeof mountPanel>;
  /** The reader's "Ask in chat": the selection as a chip, the composer focused. */
  askAbout(event: any): void;
  dispose(): void;
}

export function createPanel(opts: { id: string; version: string; win: any; dataDir: string; shadow: ShadowRoot }): Panel {
  const bundle = createHost(opts);
  const api = mountPanel(opts.shadow, bundle.host);
  const askAbout = (event: any) => {
    const chip = bundle.context.takeSelection(event); // the tracker may not have seen it: the panel was not loaded yet
    if (chip) api.addChip(chip);
  };
  return { bundle, host: bundle.host, api, askAbout, dispose() { api.dispose(); bundle.dispose(); } };
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
