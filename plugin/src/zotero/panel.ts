// The heavy half, loaded the first time the panel opens (panel.js): host, agent runtime and UI.
// plugin.js, which Zotero loads at startup, only owns the container, the button and the shortcuts.
import type { PanelHost } from "../types.ts";
import { createHost, type HostBundle } from "./host.ts";
import { mountPanel } from "../ui/index.ts";

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
