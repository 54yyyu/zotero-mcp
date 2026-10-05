// Puts the panel in Zotero's main window. Measured on Zotero 10: #tabs-deck sits in an hbox below the tab bar,
// so a sibling appended to that hbox is right of the library AND of every reader tab, and stays put when tabs switch.
import type { Panel } from "./panel.ts";

const XHTML = "http://www.w3.org/1999/xhtml";
const MIN_W = 300;
const MAX_W = 900;

export interface Injected {
  panel: any;
  toggleButton: any;
  /** Loads and mounts the panel on first use. */
  ensure(): Panel;
  /** The panel if it has been loaded; null before. */
  loaded(): Panel | null;
  show(): void;
  hide(): void;
  toggle(): void;
  isOpen(): boolean;
  remove(): void;
}

const ICON = "data:image/svg+xml;utf8," + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 128 128" fill="none" stroke="context-stroke" stroke-linecap="round" stroke-linejoin="round"><path stroke-width="10" d="M40 12H88a26 26 0 0 1 26 26V78a26 26 0 0 1-26 26H62L38 118V104H40a26 26 0 0 1-26-26V38a26 26 0 0 1 26-26Z"/><path stroke-width="12" d="M47 39H81L47 77H81"/></svg>');

export function injectPanel(win: any, load: (shadow: ShadowRoot) => Panel, prefs: { get(k: string): any; set(k: string, v: any): void; json<T>(k: string, fallback: T): T }): Injected | null {
  const doc = win.document;
  const deck = doc.getElementById("tabs-deck");
  const row = deck?.parentElement;
  if (!deck || !row) return null;

  const width = Math.min(MAX_W, Math.max(MIN_W, Number(prefs.get("width")) || 400));

  // The panel: a vbox sibling of the deck, with a drag handle on its left edge.
  const panel = doc.createXULElement("vbox");
  panel.id = "zmc-panel";
  panel.style.cssText = `flex: 0 0 auto; width: ${width}px; min-width: ${MIN_W}px; max-width: ${MAX_W}px; position: relative; overflow: hidden; border-inline-start: 1px solid var(--fill-quinary, rgba(128,128,128,.25));`;
  const surface = doc.createElementNS(XHTML, "div");
  surface.id = "zmc-root";
  surface.style.cssText = "flex: 1 1 auto; min-height: 0; width: 100%; display: flex; flex-direction: column;";
  const shadow = surface.attachShadow({ mode: "open" });
  const handle = doc.createElementNS(XHTML, "div");
  handle.id = "zmc-resize";
  handle.style.cssText = "position: absolute; inset-block: 0; inset-inline-start: -3px; width: 7px; cursor: col-resize; z-index: 10;";
  panel.append(surface, handle);
  row.appendChild(panel);

  handle.addEventListener("pointerdown", (e: PointerEvent) => {
    handle.setPointerCapture(e.pointerId);
    const startX = e.screenX, startW = panel.getBoundingClientRect().width;
    const move = (m: PointerEvent) => {
      const w = Math.min(MAX_W, Math.max(MIN_W, startW + (startX - m.screenX)));
      panel.style.width = `${w}px`;
    };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      prefs.set("width", Math.round(panel.getBoundingClientRect().width));
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });

  // The close button inside the panel (ui/ has no other way to reach us).
  surface.addEventListener("zmc-close", () => set(false));

  // The UI, agent and host load the first time the panel is shown; with the panel closed they cost nothing.
  let panelApi: Panel | null = null;
  const ensure = () => (panelApi ??= load(shadow));

  // Toolbar toggle, in the tab bar's own toolbar (right of the tabs menu).
  const toolbar = doc.getElementById("zotero-tabs-toolbar");
  const button = doc.createXULElement("toolbarbutton");
  button.id = "zmc-toggle";
  button.className = "zotero-tb-button";
  button.setAttribute("tooltiptext", "Chat");
  button.setAttribute("aria-label", "Chat");
  button.style.cssText = `-moz-context-properties: stroke; stroke: currentColor; list-style-image: url("${ICON}");`;
  toolbar?.insertBefore(button, doc.getElementById("zotero-tb-tabs-menu")?.nextSibling ?? null);

  let open = prefs.json<{ openAtStart?: boolean }>("settings", {}).openAtStart === true;
  const apply = () => {
    panel.hidden = !open;
    if (open) button.setAttribute("open", "true"); else button.removeAttribute("open");
    button.setAttribute("aria-pressed", String(open));
  };
  const set = (v: boolean) => { open = v; apply(); if (v) ensure().api.focusComposer(); };
  button.addEventListener("command", () => set(!open));
  apply();
  // Set to open at start: load once Zotero is idle, so its own startup is not delayed.
  if (open) win.requestIdleCallback(() => { if (open) ensure(); }, { timeout: 5000 });

  // Keyboard: toggle, and the prompt slots.
  const mac = Zotero.isMac;
  const onKey = (e: KeyboardEvent) => {
    const mod = mac ? e.metaKey : e.ctrlKey;
    if (mod && e.altKey && !e.shiftKey && e.code === "KeyL") { e.preventDefault(); set(!open); return; }
    const slotMod = mac ? e.metaKey && e.ctrlKey : e.ctrlKey && e.altKey;
    const m = /^Digit([1-4])$/.exec(e.code);
    if (slotMod && m) { e.preventDefault(); if (!open) set(true); ensure().api.runPrompt(Number(m[1])); }
  };
  win.addEventListener("keydown", onKey, true);

  return {
    panel, toggleButton: button, ensure, loaded: () => panelApi,
    show: () => set(true), hide: () => set(false), toggle: () => set(!open), isOpen: () => open,
    remove() {
      win.removeEventListener("keydown", onKey, true);
      panelApi?.dispose();
      button.remove();
      panel.remove();
    },
  };
}
