// How many processes a setup check and a session start cost: one detect() at a time, and a program found once is not
// looked for again until the setup is rechecked.
import assert from "node:assert/strict";
import { chmodSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { after, describe, it } from "node:test";
import { createRuntime } from "../../src/agent/index.ts";
import { findBinary, forgetBinaries } from "../../src/agent/env.ts";
import type { Spawner } from "../../src/types.ts";
import { cleanupTemp, fakeBin, fakeBridgeDir, hermeticSpawner } from "./helpers.ts";

after(cleanupTemp);

/** `spawner`, counting the helper processes it runs. */
function counting(spawner: Spawner): Spawner & { runs: number } {
  const s = { ...spawner, runs: 0, run: (...a: Parameters<Spawner["run"]>) => { s.runs++; return spawner.run(...a); } };
  return s;
}

describe("detect()", () => {
  it("asked twice at once (the setup check and the agent list), runs once", async () => {
    const one = counting(hermeticSpawner());
    await createRuntime({ spawner: one, bridgeDir: fakeBridgeDir(), nodePath: process.execPath }).detect();

    const two = counting(hermeticSpawner());
    const rt = createRuntime({ spawner: two, bridgeDir: fakeBridgeDir(), nodePath: process.execPath });
    const [a, b] = await Promise.all([rt.detect(), rt.detect()]);
    assert.deepEqual(a, b);
    assert.equal(two.runs, one.runs, "two callers cost what one does");

    await rt.detect();
    assert.ok(two.runs > one.runs, "asked again later, it looks again");
  });
});

describe("findBinary", () => {
  it("remembers where a program was found, per PATH, until the setup is rechecked", async () => {
    const bin = fakeBin({ tool: "exit 0" });
    const spawner = counting(hermeticSpawner());
    const env = { ...(await spawner.baseEnv()), PATH: `${bin}:/usr/bin:/bin` };
    assert.equal(await findBinary(spawner, env, "tool"), join(bin, "tool"));
    assert.equal(await findBinary(spawner, env, "tool"), join(bin, "tool"));
    assert.equal(spawner.runs, 1, "the second ask starts no process");

    assert.equal(await findBinary(spawner, { ...env, PATH: "/usr/bin:/bin" }, "tool"), null, "another PATH is another question");
    rmSync(join(bin, "tool"));
    forgetBinaries(spawner);
    assert.equal(await findBinary(spawner, env, "tool"), null, "after a recheck it is looked for again");
  });

  it("a miss is not remembered: a program installed since is found", async () => {
    const bin = fakeBin({});
    const spawner = hermeticSpawner();
    const env = { ...(await spawner.baseEnv()), PATH: `${bin}:/usr/bin:/bin` };
    assert.equal(await findBinary(spawner, env, "late-tool"), null);
    fakeBinInto(bin, "late-tool");
    assert.equal(await findBinary(spawner, env, "late-tool"), join(bin, "late-tool"));
  });
});

function fakeBinInto(dir: string, name: string): void {
  writeFileSync(join(dir, name), "#!/bin/sh\nexit 0\n");
  chmodSync(join(dir, name), 0o755);
}
