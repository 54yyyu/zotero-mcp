import { test } from "node:test";
import assert from "node:assert/strict";
import { AcpClient } from "../../src/agent/acp.ts";
import type { Proc, Spawner } from "../../src/types.ts";

test("an update that arrives before anyone listens (commands in the same write as the session/new reply) reaches the first listener", async () => {
  let line: (l: string) => void = () => {};
  const proc: Proc = { pid: 1, write() {}, onStdoutLine(cb) { line = cb; }, onStderr() {}, exited: new Promise(() => {}), async kill() {} };
  const spawner = { spawn: async () => proc } as unknown as Spawner;
  const client = await AcpClient.spawn({ spawner, command: "x", args: [], cwd: "/", env: {} });
  line(JSON.stringify({ jsonrpc: "2.0", method: "session/update", params: { sessionId: "s1", update: { sessionUpdate: "available_commands_update", availableCommands: [{ name: "compact" }] } } }));
  const got: [unknown, unknown][] = [];
  client.onUpdate((u, sid) => got.push([u["sessionUpdate"], sid]));
  assert.deepEqual(got, [["available_commands_update", "s1"]]);
  line(JSON.stringify({ jsonrpc: "2.0", method: "session/update", params: { sessionId: "s1", update: { sessionUpdate: "usage_update" } } }));
  assert.equal(got.length, 2, "later updates go straight through");
  const late: unknown[] = [];
  client.onUpdate((u) => late.push(u));
  assert.deepEqual(late, [], "the held ones go to the first listener only");
});
