import assert from "node:assert/strict";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { CATALOG_CLEANUP, createRuntime } from "../../src/agent/index.ts";
import type { AgentSession, ChatEvent, StartOpts } from "../../src/types.ts";
import { alive, cleanupTemp, collect, fakeBin, fakeBridgeDir, hermeticSpawner, tempDir, textOf, waitFor } from "./helpers.ts";

after(cleanupTemp);

const BRIEF = "BRIEF: you are in a Zotero panel";

interface Harness {
  session: AgentSession;
  cwd: string;
  close: () => Promise<void>;
}

const open: Harness[] = [];
after(async () => {
  await Promise.all(open.map((h) => h.close()));
});

async function start(opts: Partial<StartOpts> & { variant?: string; env?: Record<string, string> } = {}, extra: { initTimeoutMs?: number; baseEnv?: Record<string, string>; path?: string[] } = {}): Promise<Harness> {
  const cwd = tempDir("zmc-ws-");
  const rt = createRuntime({
    spawner: hermeticSpawner(extra.baseEnv ?? {}, extra.path ?? []),
    bridgeDir: fakeBridgeDir(),
    nodePath: process.execPath,
    ...(extra.initTimeoutMs ? { initTimeoutMs: extra.initTimeoutMs } : {}),
  });
  const { variant, ...rest } = opts;
  const session = await rt.start({
    backend: "claude-code",
    cwd,
    brief: BRIEF,
    ...rest,
    env: { MOCK_VARIANT: variant ?? "claude", ...(opts.env ?? {}) },
  });
  const h = { session, cwd, close: () => session.close() };
  open.push(h);
  return h;
}

const ofType = <T extends ChatEvent["t"]>(events: ChatEvent[], t: T) => events.filter((e): e is Extract<ChatEvent, { t: T }> => e.t === t);

describe("session basics", () => {
  it("reports the account, models, modes and image support from the handshake", async () => {
    const { session } = await start();
    assert.equal(session.account, "Mock Max");
    assert.equal(session.supportsImages, true);
    assert.deepEqual(session.models().map((m) => m.id), ["opus", "sonnet", "haiku"], '"default" is not a choice of its own');
    assert.equal(session.currentModel(), "opus");
    assert.deepEqual(session.modes().map((m) => m.id), ["default", "acceptEdits", "plan", "auto", "bypassPermissions"]);
    assert.equal(session.currentMode(), "auto");
    assert.match(session.sessionId, /^mock-/);
  });

  it("streams a plain reply as turn_start, text deltas in order, turn_end with usage", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "hi" });
    assert.equal(events[0]!.t, "turn_start");
    assert.deepEqual(ofType(events, "text").map((e) => e.delta), ["Hel", "lo ", "from ", "mock"]);
    const end = events.at(-1) as Extract<ChatEvent, { t: "turn_end" }>;
    assert.equal(end.t, "turn_end");
    assert.equal(end.stop, "end_turn");
    assert.deepEqual(end.usage, { inputTokens: 10, outputTokens: 5 });
    const turn = (events[0] as Extract<ChatEvent, { t: "turn_start" }>).turn;
    assert.ok(events.every((e) => !("turn" in e) || e.turn === turn), "every event of a turn carries its id");
  });

  it("does not emit what arrives outside a turn (startup banner)", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await new Promise((r) => setTimeout(r, 150));
    assert.deepEqual(events, []);
  });

  it("translates thoughts, plans, refusal and max_tokens", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:think" });
    assert.equal(ofType(events, "thought").map((e) => e.delta).join(""), "hmm ok");
    events.length = 0;
    await session.prompt({ text: "SCENARIO:plan" });
    assert.deepEqual(ofType(events, "plan")[0]!.entries, [
      { content: "Look", status: "completed" },
      { content: "Answer", status: "in_progress" },
    ]);
    events.length = 0;
    await session.prompt({ text: "SCENARIO:refuse" });
    assert.equal(ofType(events, "turn_end")[0]!.stop, "refusal");
    events.length = 0;
    await session.prompt({ text: "SCENARIO:maxtok" });
    assert.equal(ofType(events, "turn_end")[0]!.stop, "max_tokens");
  });

  it("an ephemeral session asks Claude not to persist it; a normal one does not", async () => {
    const dump = join(tempDir("zmc-new-"), "new.json");
    const { session } = await start({ ephemeral: true, env: { MOCK_NEW_DUMP: dump } });
    assert.equal(JSON.parse(readFileSync(dump, "utf8")).claudeCode.options.persistSession, false);
    await session.close();
    const { session: normal } = await start({ env: { MOCK_NEW_DUMP: dump } });
    assert.equal(JSON.parse(readFileSync(dump, "utf8")).claudeCode, undefined, "a real chat is kept so it can be resumed");
    await normal.close();
  });

  it("the probe cleanup removes the empty project folder Claude makes, and never a folder with content", async () => {
    const home = tempDir("zmc-home-");
    const spawner = hermeticSpawner({ HOME: home });
    const env = await spawner.baseEnv();
    const encode = (p: string) => p.replace(/[^A-Za-z0-9]/g, "-");
    const run = async (cwd: string) => spawner.run("/bin/sh", ["-c", CATALOG_CLEANUP, "sh", cwd], { env });

    const empty = tempDir("zmc-probe-");
    const emptyProj = join(home, ".claude", "projects", encode(realpathSync(empty)));
    mkdirSync(join(emptyProj, "memory"), { recursive: true });
    assert.equal((await run(empty)).code, 0);
    assert.ok(!existsSync(emptyProj) && !existsSync(empty), "empty project folder and temp workspace are gone");

    const used = tempDir("zmc-probe-");
    const usedProj = join(home, ".claude", "projects", encode(realpathSync(used)));
    mkdirSync(join(usedProj, "memory"), { recursive: true });
    writeFileSync(join(usedProj, "abc.jsonl"), "{}\n");
    await run(used);
    assert.ok(existsSync(join(usedProj, "abc.jsonl")), "a project folder with a session in it is left alone");
  });

  it("a turn that ends cleanly with nothing produced says so (pi reports a rejected request that way)", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:empty" });
    const notice = ofType(events, "notice").find((n) => /without answering/.test(n.message));
    assert.ok(notice && notice.level === "warn" && notice.hint, "a warning with a hint");
    assert.deepEqual(events.map((e) => e.t), ["turn_start", "notice", "turn_end"], "the warning comes before turn_end");
    events.length = 0;
    await session.prompt({ text: "hello" });
    assert.ok(!ofType(events, "notice").length, "an answered turn gets no warning");
    events.length = 0;
    await session.prompt({ text: "SCENARIO:refuse" });
    assert.ok(!ofType(events, "notice").length, "only a clean end_turn is suspect: a refusal speaks for itself");
  });

  it("refuses a second prompt while one runs", async () => {
    const { session } = await start();
    const first = session.prompt({ text: "SCENARIO:slow" });
    await assert.rejects(session.prompt({ text: "again" }), /already running/);
    await session.cancel();
    await first;
  });

  it("sends images as ACP image blocks", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "look", images: [{ mime: "image/png", data: "AAAA" }] });
    assert.match(textOf(events), /images:1/);
  });
});

describe("tools", () => {
  it("upserts tool events by id, merging fields and mapping kind/status", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:tool" });
    const tc1 = ofType(events, "tool").filter((e) => e.id === "tc1");
    assert.deepEqual(tc1.map((e) => e.status), ["pending", "running", "done"]);
    assert.ok(tc1.every((e) => e.kind === "execute"));
    assert.equal(tc1[0]!.input, undefined, "an empty rawInput is not an input");
    assert.deepEqual(tc1[1]!.input, { command: "zotero-cli search foo" });
    assert.equal(tc1[1]!.title, "`zotero-cli search foo`");
    // the last update omitted title and input: the merged state keeps them
    assert.equal(tc1[2]!.title, "`zotero-cli search foo`");
    assert.deepEqual(tc1[2]!.input, { command: "zotero-cli search foo" });
    assert.equal(tc1[2]!.output, "3 results");
    // the tool name comes from _meta.claudeCode.toolName, joined on toolCallId (extra field `name`)
    assert.ok(tc1.every((e) => (e as { name?: string }).name === "Bash"));

    const tc2 = ofType(events, "tool").filter((e) => e.id === "tc2");
    assert.deepEqual(tc2.map((e) => e.status), ["pending", "failed"]);
    assert.equal(tc2[1]!.kind, "read");
    assert.equal(tc2[1]!.output, "no such file", "toolResponse stderr stands in for missing content");
    assert.equal((tc2[1] as { name?: string }).name, undefined, "no name, no invention from the title");
    assert.equal(textOf(events), "done");
  });
});

describe("permissions", () => {
  async function ask(answer: (session: AgentSession, ev: Extract<ChatEvent, { t: "permission" }>) => void) {
    const { session } = await start();
    const { events } = collect(session);
    const run = session.prompt({ text: "SCENARIO:permit" });
    const ev = await waitFor(() => ofType(events, "permission")[0], 5000, "permission event");
    answer(session, ev);
    await run;
    return { events, ev };
  }

  it("round-trips an allow choice", async () => {
    const { events, ev } = await ask((s, e) => s.respondPermission(e.id, "allow"));
    assert.equal(ev.title, "Run rm");
    assert.equal(ev.kind, "execute");
    assert.deepEqual(ev.input, { command: "rm -rf x" });
    assert.deepEqual(ev.options, [
      { id: "allow_always", name: "Always Allow", kind: "allow_always" },
      { id: "allow", name: "Allow", kind: "allow_once" },
      { id: "reject", name: "Reject", kind: "reject_once" },
    ]);
    assert.equal((ev as { name?: string }).name, "Bash", "name joined from the earlier tool_call");
    const resolved = ofType(events, "permission").filter((e) => e.resolved !== undefined);
    assert.equal(resolved.length, 1);
    assert.equal(resolved[0]!.id, ev.id);
    assert.equal(resolved[0]!.resolved, "allow");
    assert.equal(textOf(events), "permission:allow");
  });

  it("a null answer, or an option the bridge never offered, is a refusal", async () => {
    let r = await ask((s, e) => s.respondPermission(e.id, null));
    assert.equal(textOf(r.events), "permission:cancelled");
    assert.equal(ofType(r.events, "permission").at(-1)!.resolved, "cancelled");
    r = await ask((s, e) => s.respondPermission(e.id, "allow_everything_forever"));
    assert.equal(textOf(r.events), "permission:cancelled");
  });

  it("cancel() with a question open answers it cancelled and ends the turn cancelled", async () => {
    const { session } = await start();
    const { events } = collect(session);
    const run = session.prompt({ text: "SCENARIO:permit" });
    await waitFor(() => ofType(events, "permission")[0], 5000, "permission event");
    await session.cancel();
    await run;
    assert.equal(ofType(events, "permission").at(-1)!.resolved, "cancelled");
    assert.equal(ofType(events, "turn_end")[0]!.stop, "cancelled");
  });
});

describe("cancel, errors, crashes", () => {
  it("cancel() ends a slow turn as cancelled, and the session stays usable", async () => {
    const { session } = await start();
    const { events } = collect(session);
    const run = session.prompt({ text: "SCENARIO:slow" });
    await waitFor(() => ofType(events, "text").length >= 2, 5000, "some streaming");
    await session.cancel();
    await run;
    assert.equal(ofType(events, "turn_end")[0]!.stop, "cancelled");
    assert.ok(ofType(events, "text").length < 50, "it stopped streaming");
    events.length = 0;
    await session.prompt({ text: "again" });
    assert.equal(textOf(events), "Hello from mock");
  });

  it("cancel() when idle does nothing", async () => {
    const { session } = await start();
    await session.cancel();
    const { events } = collect(session);
    await session.prompt({ text: "hi" });
    assert.equal(ofType(events, "turn_end")[0]!.stop, "end_turn");
  });

  it("an error response becomes a notice and turn_end error; prompt resolves, session lives on", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:error" });
    const notice = ofType(events, "notice")[0]!;
    assert.equal(notice.level, "error");
    assert.match(notice.message, /mock internal error/);
    assert.equal(ofType(events, "turn_end")[0]!.stop, "error");
    events.length = 0;
    await session.prompt({ text: "ok now" });
    assert.equal(textOf(events), "Hello from mock");
  });

  it("an authentication error carries a sign-in hint", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:autherror" });
    const notice = ofType(events, "notice")[0]!;
    assert.match(notice.message, /Authentication required/);
    assert.match(notice.hint ?? "", /Sign in/i);
  });

  it("bridge death mid-turn: notice error, turn_end error, prompt rejects; later prompts reject at once", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await assert.rejects(session.prompt({ text: "SCENARIO:crash" }), /exited|killed|stream/);
    assert.equal(textOf(events), "about to ");
    const notice = ofType(events, "notice")[0]!;
    assert.equal(notice.level, "error");
    assert.match(notice.message, /ended while answering/);
    assert.match(notice.message, /code 3/);
    const end = ofType(events, "turn_end");
    assert.equal(end.length, 1);
    assert.equal(end[0]!.stop, "error");
    // order: text, notice, turn_end
    assert.deepEqual(events.map((e) => e.t), ["turn_start", "text", "notice", "turn_end"]);
    const t0 = Date.now();
    await assert.rejects(session.prompt({ text: "hello?" }));
    assert.ok(Date.now() - t0 < 1000, "a request into a dead pipe rejects at once");
  });

  it("close() kills the bridge and its whole process group, and waits", async () => {
    const dump = join(tempDir(), "dump.json");
    const { session } = await start({ env: { MOCK_DUMP: dump, MOCK_GRANDCHILD: "1" } });
    const facts = JSON.parse(readFileSync(dump, "utf8")) as { pid: number; grandchild: number };
    assert.ok(alive(facts.pid) && alive(facts.grandchild));
    await session.close();
    assert.equal(alive(facts.pid), false, "bridge gone when close() resolves");
    assert.equal(alive(facts.grandchild), false, "grandchild gone too");
    await session.close(); // twice is fine
    await assert.rejects(session.prompt({ text: "x" }), /closed/);
  });
});

describe("model, mode and brief", () => {
  it("sends the model explicitly after session/new (claude: config option, the bridge picks the alias)", async () => {
    const { session } = await start({ model: "claude-sonnet-5-5" });
    assert.equal(session.currentModel(), "sonnet");
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:whoami" });
    assert.match(textOf(events), /"model":"sonnet"/);
  });

  it("keeps the bridge's default when it rejects the model", async () => {
    const { session } = await start({ model: "no-such-model" });
    assert.equal(session.currentModel(), "opus");
  });

  it("sets the mode, and setModel/setMode work later", async () => {
    const { session } = await start({ mode: "default" });
    assert.equal(session.currentMode(), "default");
    await session.setMode("plan");
    assert.equal(session.currentMode(), "plan");
    await session.setModel("haiku");
    assert.equal(session.currentModel(), "haiku");
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:whoami" });
    assert.match(textOf(events), /"mode":"plan"/);
  });

  it("ignores a mode the bridge does not offer", async () => {
    const { session } = await start({ mode: "plan-but-not-really" });
    assert.equal(session.currentMode(), "auto");
  });

  it("codex shape: models are the unique bases, set via session/set_model with the level recombined, its own modes", async () => {
    const { session } = await start({ backend: "codex", variant: "codex", model: "gpt-5.5", mode: "read-only" });
    assert.deepEqual(session.models().map((m) => m.id), ["gpt-6-sol", "gpt-6-luna", "gpt-5.5"]);
    assert.equal(session.currentModel(), "gpt-5.5");
    assert.deepEqual(session.modes().map((m) => m.id), ["read-only", "agent"]);
    assert.equal(session.currentMode(), "read-only");
  });

  it("pi shape: config-option model, and its `modes` (thinking levels) are not offered as modes", async () => {
    const { session } = await start({ backend: "pi", variant: "pi", model: "sonnet", mode: "high" });
    assert.equal(session.currentModel(), "sonnet");
    assert.deepEqual(session.modes(), []);
    assert.equal(session.currentMode(), undefined);
  });

  it("pi's startup banner, echoed late inside the first turn, is not shown", async () => {
    const { session } = await start({ backend: "pi", variant: "pi", env: { MOCK_BANNER_IN_TURN: "1" } });
    const { events } = collect(session);
    await session.prompt({ text: "hi" });
    assert.equal(textOf(events), "Hello from mock");
  });

  it("claude gets the brief as _meta.systemPrompt.append, not in the prompt", async () => {
    const { session } = await start();
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:brief" });
    assert.equal(textOf(events), `brief=${BRIEF}|prompt=SCENARIO:brief`);
  });

  it("codex/pi get the brief once, on the first prompt only", async () => {
    const { session } = await start({ backend: "codex", variant: "codex" });
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:brief" });
    const first = textOf(events);
    assert.match(first, /^brief=none\|prompt=<zotero-panel-brief>\nBRIEF: you are in a Zotero panel\n<\/zotero-panel-brief>\n\nSCENARIO:brief$/);
    events.length = 0;
    await session.prompt({ text: "SCENARIO:brief" });
    assert.equal(textOf(events), "brief=none|prompt=SCENARIO:brief");
  });
});

describe("session/load", () => {
  it("does not emit replayed history, whether it arrives before or after the response", async () => {
    for (const late of [false, true]) {
      const h = await start({ resumeSessionId: "mock-old", ...(late ? { env: { MOCK_LOAD: "late" } } : {}) });
      assert.equal(h.session.sessionId, "mock-old");
      const { events } = collect(h.session);
      await new Promise((r) => setTimeout(r, 150));
      assert.deepEqual(events, [], `no live events from replay (late=${late})`);
      await h.session.prompt({ text: "hi" });
      assert.equal(textOf(events), "Hello from mock");
      assert.ok(!JSON.stringify(events).includes("REPLAYED"));
      await h.close();
    }
  });

  it("a session the bridge cannot load rejects start() and leaves no process behind", async () => {
    const dump = join(tempDir(), "dump.json");
    await assert.rejects(start({ resumeSessionId: "missing-1", env: { MOCK_DUMP: dump } }), /not found/i);
    const { pid } = JSON.parse(readFileSync(dump, "utf8")) as { pid: number };
    await waitFor(() => !alive(pid), 3000, "bridge to be killed");
  });

  it("a resumed codex session does not get the brief again", async () => {
    const { session } = await start({ backend: "codex", variant: "codex", resumeSessionId: "mock-old" });
    const { events } = collect(session);
    await session.prompt({ text: "SCENARIO:brief" });
    assert.equal(textOf(events), "brief=none|prompt=SCENARIO:brief");
  });
});

describe("handshake failures", () => {
  it("a bridge that never answers initialize rejects within the bound and is killed", async () => {
    const dump = join(tempDir(), "dump.json");
    const t0 = Date.now();
    await assert.rejects(start({ env: { MOCK_INIT: "hang", MOCK_DUMP: dump } }, { initTimeoutMs: 300 }), /did not answer initialize within 300 ms/);
    assert.ok(Date.now() - t0 < 4000);
    const { pid } = JSON.parse(readFileSync(dump, "utf8")) as { pid: number };
    await waitFor(() => !alive(pid), 3000, "bridge to be killed");
  });

  it("a bridge that never answers session/new rejects within the bound", async () => {
    await assert.rejects(start({ env: { MOCK_NEW: "hang" } }, { initTimeoutMs: 300 }), /did not answer session\/new within 300 ms/);
  });

  it("a protocol version we do not speak is refused", async () => {
    await assert.rejects(start({ env: { MOCK_INIT: "badversion" } }), /protocolVersion 99/);
  });

  it("a node that cannot be spawned rejects, never throws uncaught", async () => {
    const rt = createRuntime({ spawner: hermeticSpawner(), bridgeDir: fakeBridgeDir(), nodePath: "/nonexistent/node" });
    await assert.rejects(rt.start({ backend: "claude-code", cwd: tempDir(), brief: "x" }), /could not start|ENOENT|not found|locate/i);
  });

  it("a missing node is a clear error", async () => {
    const rt = createRuntime({ spawner: hermeticSpawner({ PATH: "/nonexistent-bin" }), bridgeDir: fakeBridgeDir() });
    await assert.rejects(rt.start({ backend: "claude-code", cwd: tempDir(), brief: "x" }), /node was not found/);
  });
});

describe("environment", () => {
  it("cleans a parent Claude Code's session variables, sets CLAUDE_CODE_EXECUTABLE, merges the caller's env", async () => {
    const dump = join(tempDir(), "dump.json");
    const claudeDir = fakeBin({ claude: "exit 0" });
    const cwd = tempDir("zmc-ws-");
    const rt = createRuntime({
      spawner: hermeticSpawner(
        { CLAUDECODE: "1", CLAUDE_CODE_SESSION_ID: "abc", CLAUDE_CODE_ENTRYPOINT: "cli", CLAUDE_PID: "42", CLAUDE_EFFORT: "high", CLAUDE_CODE_USE_BEDROCK: "1" },
        [claudeDir],
      ),
      bridgeDir: fakeBridgeDir(),
      nodePath: process.execPath,
    });
    const s = await rt.start({ backend: "claude-code", cwd, brief: "x", env: { MOCK_DUMP: dump, ANTHROPIC_API_KEY: "sk-test" } });
    open.push({ session: s, cwd, close: () => s.close() });
    const facts = JSON.parse(readFileSync(dump, "utf8")) as { cwd: string; env: Record<string, string> };
    for (const gone of ["CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_PID", "CLAUDE_EFFORT"]) {
      assert.equal(facts.env[gone], undefined, `${gone} must not reach the bridge`);
    }
    assert.equal(facts.env["CLAUDE_CODE_USE_BEDROCK"], "1", "the user's own config survives");
    assert.equal(facts.env["CLAUDE_CODE_EXECUTABLE"], join(claudeDir, "claude"));
    assert.equal(facts.env["ANTHROPIC_API_KEY"], "sk-test");
    assert.ok(facts.env["PATH"]!.startsWith(process.execPath.replace(/\/node$/, "") + ":"), "the chosen node's dir leads PATH");
    assert.ok(facts.cwd.endsWith(cwd.split("/").pop()!), "the bridge runs in the workspace");
  });

  it("codex and pi do not get CLAUDE_CODE_EXECUTABLE", async () => {
    const dump = join(tempDir(), "dump.json");
    const claudeDir = fakeBin({ claude: "exit 0" });
    const rt = createRuntime({ spawner: hermeticSpawner({}, [claudeDir]), bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
    const cwd = tempDir("zmc-ws-");
    const s = await rt.start({ backend: "codex", cwd, brief: "x", env: { MOCK_DUMP: dump, MOCK_VARIANT: "codex" } });
    open.push({ session: s, cwd, close: () => s.close() });
    const facts = JSON.parse(readFileSync(dump, "utf8")) as { env: Record<string, string> };
    assert.equal(facts.env["CLAUDE_CODE_EXECUTABLE"], undefined);
  });
});
