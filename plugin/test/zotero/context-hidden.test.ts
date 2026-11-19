// The context tracker (zotero/context.ts) while the panel is closed: no timer, and Zotero's events (every item a sync
// touches, every tab switch, every selection) reach no listener. Shown again, listeners are told once.
// context.ts only touches Zotero at call time, so a few stand-ins are enough.
import assert from "node:assert/strict";
import { test } from "node:test";

type Notify = (event: string, type: string) => void;
const observers = new Map<string, Notify>();
const popups = new Set<(e: unknown) => void>();
(globalThis as Record<string, unknown>).Zotero = {
  Reader: {
    registerEventListener: (_t: string, cb: (e: unknown) => void) => popups.add(cb),
    unregisterEventListener: (_t: string, cb: (e: unknown) => void) => popups.delete(cb),
    getByTabID: () => null,
  },
  Notifier: {
    registerObserver: (o: { notify: Notify }) => { observers.set("n", o.notify); return "n"; },
    unregisterObserver: (id: string) => observers.delete(id),
  },
};

const { ContextTracker } = await import("../../src/zotero/context.ts");

function fakeWindow() {
  const timers = new Map<number, () => void>();
  const selects = new Map<string, () => void>();
  let next = 1;
  return {
    timers, selects,
    Zotero_Tabs: { selectedType: "library", selectedID: "zotero-pane" },
    setInterval(fn: () => void) { timers.set(next, fn); return next++; },
    clearInterval(id: number) { timers.delete(id); },
    document: { getElementById: (id: string) => ({ addEventListener: (_e: string, cb: () => void) => selects.set(id, cb), removeEventListener: () => selects.delete(id) }) },
  };
}

test("a closed panel runs no timer and hears nothing; shown again it is told once", () => {
  const win = fakeWindow();
  const tracker = new ContextTracker("chat@zotero-mcp");
  let told = 0;
  tracker.start(win);
  tracker.on(() => { told++; });
  assert.equal(win.timers.size, 1, "open: the reader's page is watched");
  observers.get("n")!("modify", "item");
  win.selects.get("zotero-items-tree")!();
  assert.equal(told, 2);

  tracker.setActive(false);
  assert.equal(win.timers.size, 0, "closed: no timer");
  told = 0;
  for (let i = 0; i < 500; i++) observers.get("n")!("modify", "item"); // a sync
  observers.get("n")!("select", "tab");
  win.selects.get("zotero-items-tree")!();
  for (const cb of popups) cb({ params: { annotation: { text: "selected words" } }, reader: { _item: {}, itemID: 1 } });
  tracker.refresh();
  assert.equal(told, 0, "closed: no listener runs");

  tracker.setActive(true);
  assert.equal(told, 1, "shown: one refresh, whatever happened meanwhile");
  assert.equal(win.timers.size, 1);
  tracker.setActive(true);
  assert.equal(told, 1);
  assert.equal(win.timers.size, 1, "never two timers");

  tracker.stop();
  assert.equal(win.timers.size, 0);
  assert.equal(observers.size, 0);
  assert.equal(popups.size, 0);
});

test("a panel loaded while closed starts no timer", () => {
  const win = fakeWindow();
  const tracker = new ContextTracker("chat@zotero-mcp");
  tracker.start(win);
  tracker.setActive(false); // inject.ts says so right after loading
  assert.equal(win.timers.size, 0);
  tracker.stop();
});
