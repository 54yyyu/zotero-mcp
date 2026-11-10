// Which processes run a Zotero on a given profile: dev.mjs refuses to start a second one (Zotero would show "A copy of
// Zotero is already open" to the user) and, when it stops, stops every one it started, including a copy Zotero re-launched
// itself as (a restart after the add-on scan gets a new pid, whose parent is gone).
import { execFileSync } from "node:child_process";
import { join } from "node:path";

/** Pids in `ps -axo pid=,command=` output whose command line names `--profile <profile>`. */
export function profilePids(ps, profile, self = process.pid) {
  const out = [];
  for (const line of ps.split("\n")) {
    const m = /^\s*(\d+)\s+(.*)$/.exec(line);
    if (!m) continue;
    const pid = Number(m[1]), cmd = m[2];
    const at = cmd.indexOf(`--profile ${profile}`);
    const end = at + `--profile ${profile}`.length;
    if (at >= 0 && (end === cmd.length || cmd[end] === " ") && pid !== self) out.push(pid);
  }
  return out;
}

/** Every process using `profile` now: by its command line, and whoever holds the profile's lock file. */
export function zoterosOn(profile) {
  const pids = new Set();
  try { for (const p of profilePids(execFileSync("ps", ["-axo", "pid=,command="], { encoding: "utf8" }), profile)) pids.add(p); } catch { /* no ps */ }
  try { for (const p of execFileSync("lsof", ["-t", join(profile, ".parentlock")], { encoding: "utf8" }).split("\n")) if (Number(p)) pids.add(Number(p)); } catch { /* none, or no lsof */ }
  pids.delete(process.pid);
  return [...pids];
}
