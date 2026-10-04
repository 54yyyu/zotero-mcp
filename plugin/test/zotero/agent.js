// The real Claude Code bridge, started from inside Zotero by the Gecko spawner: handshake only, no prompt, no tokens.
async function main(ctx) {
  const { bundle } = ctx.plugin.panel();
  const host = bundle.host;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };

  const detected = await host.runtime.detect();
  out.detected = detected.map((d) => `${d.id}: ${d.available ? "ready" : "NOT ready (" + d.reason + ")"}${d.account ? " [" + d.account + "]" : ""}`);
  check(detected.find((d) => d.id === "claude-code")?.available, "claude-code available");

  const prep = await host.prepareSession();
  const t0 = Date.now();
  const session = await host.runtime.start({ backend: "claude-code", cwd: prep.cwd, brief: prep.brief, env: prep.env, auth: "subscription" });
  out.startMs = Date.now() - t0;
  out.sessionId = session.sessionId.slice(0, 8) + "…";
  out.account = session.account;
  out.models = session.models().slice(0, 6).map((m) => m.id);
  out.modes = session.modes().map((m) => m.id);
  out.currentModel = session.currentModel();
  out.currentMode = session.currentMode();
  check(session.sessionId && session.models().length > 0 && session.modes().length > 0, "session has id, models and modes");
  check(session.supportsImages, "claude supports images (area chips)");

  // the bridge process exists while the session is open, and is gone after close()
  // only OUR bridge: the user may run other ACP bridges (meeting-buddy, Zed) that must not be counted
  const bridgePids = async () => (await bundle.spawner.run("/usr/bin/pgrep", ["-f", "zotero-chat/bridges/node_modules/@agentclientprotocol/claude-agent-acp"], { env: await bundle.spawner.baseEnv() })).stdout.trim().split("\n").filter(Boolean);
  check((await bridgePids()).length >= 1, "bridge running while the session is open");
  const tClose = Date.now();
  await session.close();
  out.closeMs = Date.now() - tClose;
  let left = await bridgePids();
  for (let i = 0; i < 25 && left.length; i++) { await ctx.sleep(200); left = await bridgePids(); }
  if (left.length) {
    const env = await bundle.spawner.baseEnv();
    const ps = await bundle.spawner.run("/bin/ps", ["-o", "pid,ppid,etime,stat,command", "-p", left.join(",")], { env });
    throw new Error("FAILED: bridge still alive 5s after close(): " + ps.stdout);
  }
  out.exitedAfterMs = Date.now() - tClose;
  out.cleanClose = true;
  return out;
}
