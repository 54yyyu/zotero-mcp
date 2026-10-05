// Zotero's Settings window has a pane per plugin; ours shows the panel's own settings screen (settings.ts), the same
// cards from the same code, in a shadow root so neither stylesheet reaches the other. Zotero titles the pane; the cards
// follow the theme and the look, and a change made in an open panel shows here at once (and the other way round).
import type { BackendId, BackendStatus, SettingsHost } from "../types.ts";
import { Look } from "./appearance.ts";
import { clear, h, initEnv } from "./dom.ts";
import { settingsView } from "./settings.ts";
import { STYLES } from "./styles.ts";

/** The pane sits in Zotero's own scrolling page: natural height, and Zotero's window is the backdrop (no gradient or picture of ours). */
const PANE_STYLES = `
:host { height: auto; }
.zmc.zmc--pane { height: auto; overflow: visible; background: transparent; }
.zmc.zmc--pane::before, .zmc.zmc--pane::after { display: none; }
.zmc--pane .set { padding: var(--s1) 0 var(--s4); }
`;

export function mountSettings(root: ShadowRoot, host: SettingsHost, catalogReady: (b: BackendId) => boolean): { dispose(): void } {
  initEnv(root);
  const app = h("div.zmc.zmc--pane", { dataset: { theme: host.theme(), view: "settings" } });
  const look = new Look(app, host);
  look.apply();
  let statuses: BackendStatus[] | null = null;
  let image = host.getSettings().appearance.image;
  const view = settingsView(host, {
    statuses: () => statuses,
    refreshStatuses: () => void host.runtime.detect().then((s) => { statuses = s; view.render(); }, () => {}),
    changed: () => {},
    look, focusPrompts: false, catalogReady,
  });
  app.append(view.el);
  root.append(h("style", { text: STYLES + PANE_STYLES }), app);
  const offs = [
    host.onThemeChange(() => { app.dataset.theme = host.theme(); }),
    host.onSettingsChange(() => {
      const now = host.getSettings().appearance.image;
      if (now !== image) { image = now; look.setImage(null); } else look.apply();
      view.render();
    }),
  ];
  return { dispose() { for (const off of offs) off(); clear(root); } };
}
