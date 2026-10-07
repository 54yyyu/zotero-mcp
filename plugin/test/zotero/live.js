// Real agents through the real runtime, started from inside Zotero: Claude Code, Codex and pi (on the local Qwen).
// Spends a few hundred tokens each on the user's subscription (pi/Qwen is a local endpoint), so it only runs with
// ZMC_LIVE=1, on the cheapest model of each. ZMC_LIVE_BACKENDS=claude-code,codex,pi picks which (default: all three).
const SETUP = {
  "claude-code": { model: "haiku", auth: "subscription" },
  codex: { model: "gpt-6-luna[low]", auth: "subscription", mode: "read-only" },
  pi: { model: "qwen38/qwen38-27b", auth: "api-key" },
};
// pi reads its providers from its agent dir. ZMC_PI_AGENT_DIR points it at a scratch one (e.g. when ~/.pi/agent has a stale endpoint).

async function main(ctx) {
  if (!ctx.env("ZMC_LIVE")) return { skipped: "set ZMC_LIVE=1 to run (spends tokens)" };
  const { host } = ctx.plugin.panel();
  const wanted = (ctx.env("ZMC_LIVE_BACKENDS") || "claude-code,codex,pi").split(",");
  const out = {}, failures = [];

  for (const backend of wanted) {
    const r = out[backend] = {};
    try {
      const prep = await host.prepareSession();
      const piDir = backend === "pi" && ctx.env("ZMC_PI_AGENT_DIR");
      const env = piDir ? { ...prep.env, PI_CODING_AGENT_DIR: piDir } : prep.env;
      const session = await host.runtime.start({ backend, cwd: prep.cwd, brief: prep.brief, env, ...SETUP[backend] });
      r.account = session.account ?? "(none)";
      r.model = session.currentModel();
      r.modes = session.modes().map((m) => m.id).join(",") || "(none)";
      const events = [];
      session.on((e) => events.push(e));
      const ask = async (text) => { events.length = 0; const t0 = Date.now(); await session.prompt({ text }); return { evs: events.slice(), ms: Date.now() - t0 }; };
      const said = (evs) => evs.filter((e) => e.t === "text").map((e) => e.delta).join("");
      const must = (cond, msg) => { if (!cond) throw new Error(msg); };

      // 1. a turn streams: turn_start .. turn_end(end_turn), and the answer is right
      const t1 = await ask("Reply with exactly the single word: pong");
      r.pong = said(t1.evs).trim().slice(0, 60);
      r.firstTurnMs = t1.ms;
      r.events1 = t1.evs.map((e) => e.t + (e.t === "notice" ? `(${e.level}: ${e.message.slice(0, 160)})` : e.t === "turn_end" ? `(${e.stop})` : "")).join(" ");
      must(/pong/i.test(r.pong), "answered pong, got: " + r.pong);
      must(t1.evs[0].t === "turn_start" && t1.evs.at(-1).t === "turn_end" && t1.evs.at(-1).stop === "end_turn", "event order: " + t1.evs.map((e) => e.t).join(","));
      r.textChunks = t1.evs.filter((e) => e.t === "text").length;

      // 2. the brief reached the model (Claude: system prompt; Codex and pi: prefixed to the first prompt)
      const t2 = await ask("In one word, which shell command should you use for my Zotero library? Answer with the command name only.");
      r.brief = said(t2.evs).trim().slice(0, 60);
      must(/zotero-cli/i.test(r.brief), "brief reached the model, it said: " + r.brief);

      // 3. the context block is understood as the user's focus
      const t3 = await ask('<zotero-context>\nReading in the Zotero reader: "Are Emily and Greg more employable than Lakisha and Jamal?" · item UBWFVP8V · PDF attachment R5HINSHA · on p.2\n</zotero-context>\nWhat page am I on? Answer with just the number.');
      r.focus = said(t3.evs).trim().slice(0, 60);
      must(/\b2\b/.test(r.focus), "read the page from the context block, said: " + r.focus);
      await session.close();
    } catch (e) {
      r.FAILED = String(e).slice(0, 300);
      failures.push(backend);
    }
  }
  if (failures.length) throw new Error("live test failed for: " + failures.join(", ") + "\n" + JSON.stringify(out, null, 1));
  return out;
}
