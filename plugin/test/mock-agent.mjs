#!/usr/bin/env node
// A deterministic fake ACP agent: stdio, newline-delimited JSON-RPC 2.0. No model, no network.
// Behaviour is driven by words in the prompt text (see `scenario` below). Wire shapes follow the
// real bridges as measured on 2026-10-04 (claude-agent-acp 0.85.1, codex-acp 2.1.1).
//
// Environment:
//   MOCK_VARIANT=claude (default)  models are a config option; session/set_model -> -32601
//   MOCK_VARIANT=codex             `models` block + session/set_model; set_config_option -> -32601
//   MOCK_VARIANT=pi                config options only, `modes` are thinking levels, no permissions
//   MOCK_CODEX_HIDE_DEFAULT=1      codex: the current model (gpt-6-sol) is missing from availableModels, as on the real bridge
//   MOCK_INIT=hang                 never answer initialize (bad-handshake timeout)
//   MOCK_INIT=badversion           answer initialize with protocolVersion 99
//   MOCK_NEW=hang                  never answer session/new
//   MOCK_LOAD=late                 replay history AFTER the session/load response (some bridges do)
//   MOCK_DUMP=<file>               write argv/cwd/pid/env facts to this file at startup (env-cleaning tests)
//   MOCK_GRANDCHILD=1              also start a `sleep` grandchild in the same process group (its pid goes in the dump)
import { spawn as spawnChild } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

const variant = process.env.MOCK_VARIANT ?? "claude";
const BANNER = "STARTUP BANNER (must not be shown)";

if (process.env.MOCK_DUMP) {
  const grandchild = process.env.MOCK_GRANDCHILD ? spawnChild("sleep", ["300"], { stdio: "ignore" }).pid : undefined;
  const keys = Object.keys(process.env).filter((k) => k.startsWith("CLAUDE") || ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"].includes(k) || ["PATH", "MOCK_VARIANT", "CODEX_CONFIG", "PI_ACP_PI_COMMAND", "MAX_THINKING_TOKENS", "ANTHROPIC_MODEL"].includes(k));
  writeFileSync(process.env.MOCK_DUMP, JSON.stringify({ pid: process.pid, grandchild, argv: process.argv.slice(2), cwd: process.cwd(), env: Object.fromEntries(keys.map((k) => [k, process.env[k]])) }));
}

const MODES = [
  { id: "default", name: "Manual", description: "Always ask before making changes" },
  { id: "acceptEdits", name: "Accept edits", description: "Automatically accept all file edits" },
  { id: "plan", name: "Plan", description: "Create a plan before making changes" },
  { id: "auto", name: "Auto", description: "Claude handles permission decisions" },
  { id: "bypassPermissions", name: "Bypass permissions", description: "Accepts all permissions" },
];
const CODEX_MODES = [
  { id: "read-only", name: "Read-only", description: "Requires approval to edit files" },
  { id: "agent", name: "Auto review", description: "Only ask for risky actions" },
];
const PI_LEVELS = ["off", "low", "high"].map((id) => ({ id, name: `Thinking: ${id}`, description: null }));
const MODEL_OPTS = [
  { value: "default", name: "Default (recommended)", description: "Opus 5.5" },
  { value: "opus", name: "Opus 5.5", description: "For complex work" },
  { value: "sonnet", name: "Sonnet 5.5", description: "Most efficient" },
  { value: "haiku", name: "Haiku 4.5", description: "Fastest" },
];
const LEVELS = { "gpt-6-sol": ["low", "medium", "high"], "gpt-6-luna": ["low", "medium", "high", "max"], "gpt-5.5": ["low", "medium"] };
const BASE_NAMES = { "gpt-6-sol": "6 Sol", "gpt-6-luna": "6 Luna", "gpt-5.5": "5.5" };
const CODEX_MODELS = Object.entries(LEVELS).flatMap(([base, levels]) =>
  levels.map((l) => ({ modelId: `${base}[${l}]`, name: `${BASE_NAMES[base]} (${l})`, description: `Model ${base}. Reasoning ${l}` })),
);
const EFFORT_OPTS = ["default", "low", "medium", "high", "xhigh", "max"].map((v) => ({ value: v, name: v }));

/** sessionId -> state */
const sessions = new Map();
let nextSession = 1;
let nextRequestId = 1000;
let initParams = null;
const waiting = new Map(); // our request id -> resolve
const cancelled = new Set();

function send(frame) {
  process.stdout.write(JSON.stringify({ jsonrpc: "2.0", ...frame }) + "\n");
}
function update(sessionId, u) {
  send({ method: "session/update", params: { sessionId, update: u } });
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function configOptions(s) {
  if (variant === "pi") {
    return [
      { id: "model", name: "Model", category: "model", type: "select", currentValue: s.model, options: MODEL_OPTS.slice(1) },
      { id: "thought_level", name: "Thinking", category: "thought_level", type: "select", currentValue: s.effort, options: PI_LEVELS.map((m) => ({ value: m.id, name: m.name })) },
    ];
  }
  const modes = variant === "codex" ? CODEX_MODES : MODES;
  return [
    { id: "mode", name: "Mode", category: "mode", type: "select", currentValue: s.mode, options: modes.map((m) => ({ value: m.id, name: m.name, description: m.description })) },
    { id: "model", name: "Model", category: "model", type: "select", currentValue: s.model, options: variant === "codex" ? Object.keys(LEVELS).map((v) => ({ value: v, name: v })) : MODEL_OPTS },
    // codex has no effort option; haiku has none either (like the real bridge's "Available effort levels for this model")
    ...(variant === "claude" && s.model !== "haiku" ? [{ id: "effort", name: "Effort", category: "thought_level", type: "select", currentValue: s.effort, options: EFFORT_OPTS }] : []),
  ];
}

function sessionResult(id, s) {
  const result = { sessionId: id, configOptions: configOptions(s) };
  if (variant === "claude") result.modes = { currentModeId: s.mode, availableModes: MODES };
  if (variant === "codex") {
    result.modes = { currentModeId: s.mode, availableModes: CODEX_MODES };
    result.models = { currentModelId: s.model, availableModels: process.env.MOCK_CODEX_HIDE_DEFAULT ? CODEX_MODELS.filter((m) => !m.modelId.startsWith("gpt-6-sol")) : CODEX_MODELS };
  }
  if (variant === "pi") {
    result.modes = { currentModeId: "high", availableModes: PI_LEVELS };
    result._meta = { piAcp: { startupInfo: BANNER } };
  }
  return result;
}

const initialModel = () => (variant === "codex" ? "gpt-6-sol[medium]" : (variant === "claude" && process.env.ANTHROPIC_MODEL) || "opus"); // claude-agent-acp starts on ANTHROPIC_MODEL
const initialEffort = () => (variant === "pi" ? "high" : "medium");
const initialMode = () => (variant === "codex" ? "agent" : "auto");

async function handle(msg) {
  const { id, method, params } = msg;
  const reply = (result) => send({ id, result });
  const fail = (code, message, data) => send({ id, error: { code, message, ...(data ? { data } : {}) } });

  switch (method) {
    case "initialize":
      initParams = params;
      if (process.env.MOCK_INIT === "hang") return;
      return reply({
        protocolVersion: process.env.MOCK_INIT === "badversion" ? 99 : 1,
        agentCapabilities: { loadSession: true, promptCapabilities: { image: variant !== "pi" || true, embeddedContext: true } },
        agentInfo: { name: "mock-acp", version: "0.0.0" },
        authMethods: [],
      });

    case "session/new": {
      if (process.env.MOCK_NEW === "hang") return;
      if (process.env.MOCK_NEW_DUMP) writeFileSync(process.env.MOCK_NEW_DUMP, JSON.stringify(params?._meta ?? null));
      const sid = `mock-${nextSession++}`;
      const sp = params?._meta?.systemPrompt; // a string replaces the system prompt (a locked session), { append } adds to it
      const s = { brief: (typeof sp === "string" ? sp : sp?.append) ?? null, locked: params?._meta?.claudeCode?.options?.tools?.length === 0, model: initialModel(), effort: initialEffort(), mode: initialMode(), history: [], cwd: params?.cwd };
      sessions.set(sid, s);
      send({ method: "_auth/status_update", params: { authStatus: { kind: "account", label: "Mock Max" } } });
      // The real bridges announce commands (and pi a banner) right after session/new: no turn is running.
      reply(sessionResult(sid, s));
      update(sid, { sessionUpdate: "available_commands_update", availableCommands: [{ name: "compact", description: "Clear conversation history but keep a summary in context", input: null }] });
      update(sid, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: BANNER } });
      return;
    }

    case "session/load": {
      const sid = params?.sessionId;
      if (!sid || String(sid).startsWith("missing")) return fail(-32002, "Resource not found: session " + sid);
      const s = sessions.get(sid) ?? { brief: params?._meta?.systemPrompt?.append ?? null, model: initialModel(), effort: initialEffort(), mode: initialMode(), history: ["earlier question", "earlier answer"], cwd: params?.cwd };
      sessions.set(sid, s);
      send({ method: "_auth/status_update", params: { authStatus: { kind: "account", label: "Mock Max" } } });
      const replay = () => {
        update(sid, { sessionUpdate: "user_message_chunk", content: { type: "text", text: "REPLAYED user" } });
        update(sid, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "REPLAYED assistant" } });
        update(sid, { sessionUpdate: "tool_call", toolCallId: "old1", title: "REPLAYED tool", kind: "read", status: "completed" });
      };
      // History replay, before the response, as the spec says (MOCK_LOAD=late: after, as a straggler).
      if (process.env.MOCK_LOAD !== "late") replay();
      reply(sessionResult(sid, s));
      if (process.env.MOCK_LOAD === "late") replay();
      return;
    }

    case "session/set_model": {
      if (variant !== "codex") return fail(-32601, `"Method not found": ${method}`, { method });
      const s = sessions.get(params.sessionId);
      if (!CODEX_MODELS.some((m) => m.modelId === params.modelId)) return fail(-32602, "Invalid params: unknown model " + params.modelId);
      s.model = params.modelId;
      return reply({});
    }

    case "session/set_config_option": {
      if (variant === "codex") return fail(-32601, `"Method not found": ${method}`, { method });
      const s = sessions.get(params.sessionId);
      if (params.configId === "model") {
        // like the real bridge: a full id lands on the matching alias option
        s.model = params.value === "claude-sonnet-5-5" ? "sonnet" : params.value;
        if (!MODEL_OPTS.some((o) => o.value === s.model)) return fail(-32602, "Invalid params: unknown model " + params.value);
        if (variant === "claude") s.effort = "high"; // like the real bridge: a model change resets the effort
      } else if (params.configId === "effort" || params.configId === "thought_level") {
        if (!configOptions(s).some((o) => o.id === params.configId && o.options.some((x) => x.value === params.value))) return fail(-32602, "Invalid params: effort " + params.value);
        s.effort = params.value;
      }
      return reply({ configOptions: configOptions(s) });
    }

    case "session/set_mode": {
      const s = sessions.get(params.sessionId);
      s.mode = params.modeId;
      update(params.sessionId, { sessionUpdate: "current_mode_update", currentModeId: params.modeId });
      return reply({});
    }

    case "session/prompt":
      return prompt(id, params);

    default:
      return fail(-32601, `"Method not found": ${method}`, { method });
  }
}

async function prompt(id, params) {
  const sid = params.sessionId;
  const s = sessions.get(sid);
  const blocks = params.prompt ?? [];
  const text = blocks.filter((b) => b.type === "text").map((b) => b.text).join("\n");
  const images = blocks.filter((b) => b.type === "image").length;
  const chunk = (t) => update(sid, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: t } });
  // Context fill, as all three real bridges report it before the prompt resolves: a usage_update with used/size.
  const end = (stopReason = "end_turn") => {
    if (s) { s.used = (s.used ?? 0) + 1000 + Math.ceil(text.length / 4); update(sid, { sessionUpdate: "usage_update", used: s.used, size: 200000 }); }
    send({ id, result: { stopReason, usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } } });
  };
  cancelled.delete(sid);
  // MOCK_BANNER_IN_TURN=1 (pi): the startup banner shows up again inside the first turn, as a straggler.
  if (process.env.MOCK_BANNER_IN_TURN && s && !s.bannered) {
    s.bannered = true;
    chunk(BANNER);
  }

  if (text.includes("SCENARIO:error")) return send({ id, error: { code: -32603, message: "mock internal error" } });
  if (text.includes("SCENARIO:autherror")) return send({ id, error: { code: -32000, message: "Authentication required" } });
  if (text.includes("SCENARIO:crash")) {
    chunk("about to ");
    await sleep(20);
    process.exit(3);
  }
  if (text.includes("SCENARIO:slow")) {
    for (let i = 0; i < 200; i++) {
      if (cancelled.has(sid)) return end("cancelled");
      chunk(`tick${i} `);
      await sleep(25);
    }
    return end();
  }
  if (text.includes("SCENARIO:tool")) {
    const meta = { claudeCode: { toolName: "Bash" } };
    update(sid, { sessionUpdate: "tool_call", toolCallId: "tc1", title: "Terminal", kind: "execute", status: "pending", rawInput: {}, _meta: meta });
    update(sid, { sessionUpdate: "tool_call_update", toolCallId: "tc1", title: "`zotero-cli search foo`", status: "in_progress", rawInput: { command: "zotero-cli search foo" }, _meta: meta });
    update(sid, { sessionUpdate: "tool_call_update", toolCallId: "tc1", status: "completed", content: [{ type: "content", content: { type: "text", text: "3 results" } }] });
    update(sid, { sessionUpdate: "tool_call", toolCallId: "tc2", title: "Read paper.pdf", kind: "read", status: "pending", rawInput: { file_path: "/x/paper.pdf" } });
    update(sid, { sessionUpdate: "tool_call_update", toolCallId: "tc2", status: "failed", _meta: { claudeCode: { toolResponse: { stdout: "", stderr: "no such file" } } } });
    chunk("done");
    return end();
  }
  if (text.includes("SCENARIO:permit")) {
    update(sid, { sessionUpdate: "tool_call", toolCallId: "tc9", title: "Run rm", kind: "execute", status: "pending", rawInput: { command: "rm -rf x" }, _meta: { claudeCode: { toolName: "Bash" } } });
    const rid = nextRequestId++;
    const answer = new Promise((resolve) => waiting.set(rid, resolve));
    send({
      id: rid,
      method: "session/request_permission",
      params: {
        sessionId: sid,
        toolCall: { toolCallId: "tc9", title: "Run rm", kind: "execute", rawInput: { command: "rm -rf x" } },
        options: [
          { optionId: "allow_always", name: "Always Allow", kind: "allow_always" },
          { optionId: "allow", name: "Allow", kind: "allow_once" },
          { optionId: "reject", name: "Reject", kind: "reject_once" },
        ],
      },
    });
    const outcome = await answer;
    chunk(`permission:${outcome?.outcome?.outcome === "selected" ? outcome.outcome.optionId : "cancelled"}`);
    return end(cancelled.has(sid) ? "cancelled" : "end_turn");
  }
  if (text.includes("SCENARIO:plan")) {
    update(sid, { sessionUpdate: "plan", entries: [{ content: "Look", status: "completed", priority: "high" }, { content: "Answer", status: "in_progress", priority: "medium" }] });
    chunk("planned");
    return end();
  }
  if (text.includes("SCENARIO:states")) { // the working line's states, each held 1.5 s; the permission card waits for the test
    const hold = () => sleep(1500);
    const meta = { claudeCode: { toolName: "Bash" } };
    update(sid, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "Which audit studies are in the library?" } });
    await hold();
    for (const [tc, cmd] of [["s1", 'zotero-cli --json search "hiring audit" --limit 5'], ["s2", "zotero-cli --json read BM2004AB --start-page 7"], ["s3", "python3 tally.py"]]) {
      update(sid, { sessionUpdate: "tool_call", toolCallId: tc, title: `\`${cmd}\``, kind: "execute", status: "in_progress", rawInput: { command: cmd }, _meta: meta });
      await hold();
      update(sid, { sessionUpdate: "tool_call_update", toolCallId: tc, status: "completed", content: [{ type: "content", content: { type: "text", text: "ok" } }] });
    }
    const rid = nextRequestId++;
    const answer = new Promise((resolve) => waiting.set(rid, resolve));
    send({ id: rid, method: "session/request_permission", params: { sessionId: sid, toolCall: { toolCallId: "s4", title: "Create a note", kind: "edit", rawInput: { command: "zotero-cli notes create BM2004AB --text hi" } },
      options: [{ optionId: "allow", name: "Allow", kind: "allow_once" }, { optionId: "reject", name: "Reject", kind: "reject_once" }] } });
    await answer;
    for (const w of "Callbacks for White-sounding names were about 50% higher. ".split(" ")) { if (cancelled.has(sid)) return end("cancelled"); chunk(w + " "); await sleep(150); }
    await hold();
    chunk("Done.");
    return end();
  }
  if (text.includes("SCENARIO:hold")) { // thinking until Stop (at most a minute): the working line's cost is measured meanwhile
    update(sid, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "Thinking it over." } });
    for (let i = 0; i < 600 && !cancelled.has(sid); i++) await sleep(100);
    return end("cancelled");
  }
  if (text.includes("SCENARIO:think")) {
    update(sid, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "hmm " } });
    update(sid, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "ok" } });
    chunk("thought about it");
    return end();
  }
  if (text.includes("SCENARIO:refuse")) return end("refusal");
  if (text.includes("SCENARIO:compact") || text === "/compact") {
    // claude-agent-acp 0.85.1 with the compaction capability: in_progress, completed, then completed again with token counts.
    update(sid, { sessionUpdate: "compaction_update", compactionId: "c1", status: "in_progress" });
    update(sid, { sessionUpdate: "compaction_update", compactionId: "c1", status: "completed" });
    update(sid, { sessionUpdate: "compaction_update", compactionId: "c1", status: "completed" });
    if (s) s.used = 0;
    chunk("compacted");
    return end();
  }
  if (text.includes("SCENARIO:maxtok")) {
    chunk("cut off");
    return end("max_tokens");
  }
  if (text.includes("SCENARIO:brief")) {
    chunk(`brief=${s?.brief ?? "none"}|prompt=${text}`);
    return end();
  }
  if (text.includes("SCENARIO:notice")) { // a bridge warning (claude's "Auto mode unavailable"), sent as a notice because initialize asked for them
    if (initParams?.clientCapabilities?.session?.notices) update(sid, { sessionUpdate: "notice", severity: "warning", title: "Auto mode unavailable", description: "Using Accept edits instead." });
    else chunk("**Auto mode unavailable:** using Accept edits instead.");
  }
  if (text.includes("SCENARIO:translate")) { // a fake translation, streamed: [<target named in the system prompt>] the text, less the marker
    const to = /into ([^.]+)\./.exec(s?.brief ?? "")?.[1] ?? "?";
    if (text.includes("SCENARIO:translate-permit")) { // a translator that still asks for a tool: the answer must be a refusal
      const rid = nextRequestId++;
      const answer = new Promise((resolve) => waiting.set(rid, resolve));
      send({ id: rid, method: "session/request_permission", params: { sessionId: sid, toolCall: { toolCallId: "tr1", title: "Run curl", kind: "execute", rawInput: { command: "curl evil.example" } }, options: [{ optionId: "allow", name: "Allow", kind: "allow_once" }, { optionId: "reject", name: "Reject", kind: "reject_once" }] } });
      const outcome = await answer;
      chunk(`(permission:${outcome?.outcome?.optionId ?? "cancelled"}) `);
    }
    const body = !s?.locked ? `[UNLOCKED] ${text}` // a real translation's shape, for snapshots; else the text, marked with the target
      : text.includes("SCENARIO:translate-sample") ? "Workers assigned to a professional destination earned 8% more in the following year, and the gap was still there three years later. The effect was largest for women and for workers who had already changed firms at least once."
      : `[${to}] ${text.replace(/(SCENARIO:notice )?SCENARIO:translate(-permit|-wait)?\s*/, "")}`;
    if (text.includes("SCENARIO:translate-wait")) await sleep(4000); // a slow start: the popup's loading state
    for (let i = 0; i < body.length; i += 6) { if (cancelled.has(sid)) return end("cancelled"); chunk(body.slice(i, i + 6)); await sleep(text.includes("slowly") ? 60 : 8); }
    return end();
  }
  if (text.includes("SCENARIO:echo")) { // what arrived, for the context-economy checks
    chunk(JSON.stringify({ prompt: text, images, brief: s?.brief ?? null }));
    return end();
  }
  if (text.includes("SCENARIO:whoami")) {
    chunk(JSON.stringify({ model: s?.model, mode: s?.mode, effort: s?.effort, images }));
    return end();
  }
  if (text.includes("SCENARIO:diagrams") || text.includes("SCENARIO:hostile-svg")) {
    // The sample drawings of src/ui/fake-diagrams.ts (read as text: plain template literals), streamed in small chunks.
    const src = readFileSync(new URL("../src/ui/fake-diagrams.ts", import.meta.url), "utf8");
    const svg = (name) => new RegExp(`export const ${name} = \`([^\`]*)\``).exec(src)?.[1] ?? "";
    const fence = (name) => "```svg\n" + svg(name) + "\n```\n\n";
    const md = text.includes("SCENARIO:hostile-svg") ? fence("HOSTILE") + "After the drawing." : `How it works:\n\n${fence("PIPELINE")}Where the designs sit:\n\n${fence("MATRIX")}${fence("CAUSAL")}Done.`;
    for (let i = 0; i < md.length; i += 400) chunk(md.slice(i, i + 400));
    return end();
  }
  // Default: a plain reply streamed in chunks.
  if (text.includes("SCENARIO:empty")) return end();
  if (text.includes("SCENARIO:rich")) {
    // A realistic answer: a tool step, then Markdown with citations to ATT=<key> (an open-pdf link per page), code and math.
    const att = /ATT=([A-Z0-9]{8})/.exec(text)?.[1] ?? "ABCD1234";
    const cite = (label, page) => `[${label}, p.${page}](zotero://open-pdf/library/items/${att}?page=${page})`;
    update(sid, { sessionUpdate: "tool_call", toolCallId: "r1", title: "`zotero-cli search discrimination`", kind: "search", status: "completed", rawInput: { command: "zotero-cli search discrimination" } });
    const md = [
      "## What the paper finds\n\n",
      `White-sounding names got **50% more callbacks** than African-American-sounding names ${cite("Bertrand and Mullainathan 2004", 3)}. `,
      `The gap holds across cities and job types ${cite("Bertrand and Mullainathan 2004", 1)}. More at [the Zotero site](https://www.zotero.org) and [a bad link](javascript:alert(1)).\n\n`,
      "- The ratio is $1.50$ in the full sample\n- It is $1.22$ in sales jobs, the smallest\n\n",
      "The relationship is $$\\text{callback} = \\beta_0 + \\beta_1 \\cdot \\text{white name} + \\varepsilon$$\n\n",
      "```python\nratio = 9.65 / 6.45  # 1.50\n```\n\n",
      "| Sample | Ratio |\n|---|---|\n| All | 1.50 |\n| Sales | 1.22 |\n",
    ].join("");
    for (let i = 0; i < md.length; i += 24) { chunk(md.slice(i, i + 24)); await sleep(2); }
    return end();
  }
  for (const part of ["Hel", "lo ", "from ", "mock"]) {
    chunk(part);
    await sleep(2);
  }
  if (images) chunk(` images:${images}`);
  return end();
}

let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i);
    buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    const msg = JSON.parse(line);
    if (msg.method !== undefined && msg.id !== undefined) void handle(msg);
    else if (msg.method === "session/cancel") {
      cancelled.add(msg.params.sessionId);
    } else if (msg.id !== undefined && waiting.has(msg.id)) {
      waiting.get(msg.id)(msg.result ?? { outcome: { outcome: "cancelled" } });
      waiting.delete(msg.id);
    }
  }
});
process.stdin.on("end", () => process.exit(0));
