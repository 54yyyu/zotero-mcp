// Harness hook: when ZMC_TEST_SCRIPT is set, load that script, run its main(ctx), write the outcome, quit.
import { snapshot } from "./snapshot.ts";

export async function maybeRunTestScript(plugin: any, win: any): Promise<void> {
  const path = Services.env.get("ZMC_TEST_SCRIPT");
  if (!path) return;
  const outPath = Services.env.get("ZMC_TEST_OUT");
  const shots = Services.env.get("ZMC_SNAPSHOT_DIR");
  const logs: string[] = [];
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const ctx = {
    Zotero, Services, IOUtils, PathUtils, win, plugin,
    sleep,
    /** A realistic window: the harness default is small enough to squeeze Zotero's own panes. */
    resize: async (w: number, h: number) => { win.resizeTo(w, h); await sleep(600); },
    // Appended to <out>.log as it happens, so a script that hangs still shows how far it got.
    log: (...a: unknown[]) => {
      const line = a.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join(" ");
      logs.push(line);
      void IOUtils.writeUTF8(outPath + ".log", line + "\n", { mode: "appendOrCreate" }).catch(() => {});
    },
    async waitFor<T>(fn: () => T | Promise<T>, what = "condition", ms = 15000): Promise<T> {
      const end = Date.now() + ms;
      for (;;) {
        const v = await fn();
        if (v) return v;
        if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
        await sleep(100);
      }
    },
    snapshot: (name: string) => snapshot(win, PathUtils.join(shots, `${name}.png`)),
    env: (k: string) => Services.env.get(k),
  };
  let result: unknown, error: string | null = null;
  try {
    const scope: any = { ...ctx, console: { log: ctx.log } };
    Services.scriptloader.loadSubScript(PathUtils.toFileURI(path), scope);
    result = await scope.main(ctx);
  } catch (e: any) {
    error = `${e}${e?.stack ? "\n" + e.stack : ""}`;
  }
  await IOUtils.writeUTF8(outPath, JSON.stringify({ ok: !error, result, error, logs }, null, 2));
  if (!Services.env.get("ZMC_KEEP")) Services.startup.quit(Ci.nsIAppStartup.eForceQuit);
}
