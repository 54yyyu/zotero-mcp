// The agent's working directory: created, with the zotero-cli skill installed into it.
//   zotero-mcp install-skill --target claude --target agents --root <dir>
// The installer (src/zotero_mcp/skill_install.py) is idempotent and never clobbers: a destination
// that exists and differs is reported and left alone (exit 1 when nothing was written at all), and
// in AGENTS.md only the block between its own markers is ever touched.
// No node:fs (Gecko): the directory is made by /bin/sh.
import type { Spawner } from "../types.ts";
import { dirnameOf, findBinary } from "./env.ts";

export interface WorkspaceOpts {
  /** Absolute path of `zotero-cli` when the host found it; `zotero-mcp` is looked for beside it first. */
  zoteroCli?: string;
  /** The folder is the panel's own: bring the installed skill up to the packaged one (a new version adds commands, and an old copy would hide them). Anyone else's folder keeps the never-clobber rule. */
  refresh?: boolean;
}

/** The ways to run `zotero-mcp`, best first. */
async function installerCommands(spawner: Spawner, env: Record<string, string>, opts: WorkspaceOpts): Promise<{ command: string; args: string[] }[]> {
  const out: { command: string; args: string[] }[] = [];
  const add = (command: string | null, args: string[] = []) => {
    if (command && !out.some((o) => o.command === command && o.args.join(" ") === args.join(" "))) out.push({ command, args });
  };
  if (opts.zoteroCli) {
    const sibling = `${dirnameOf(opts.zoteroCli)}/zotero-mcp`;
    const r = await spawner.run("/bin/sh", ["-c", '[ -x "$1" ] && echo ok', "sh", sibling], { env, timeoutMs: 5_000 }).catch(() => null);
    if (r?.stdout.trim() === "ok") add(sibling);
  }
  add(await findBinary(spawner, env, "zotero-mcp"));
  const uv = await findBinary(spawner, env, "uv");
  if (uv) add(uv, ["tool", "run", "--from", "zotero-mcp-server", "zotero-mcp"]);
  return out;
}

/**
 * Create the chat workspace and put the skill in it. Returns the directory to use as the agent's cwd.
 * A missing or failing installer is not fatal (the brief tells the agent to use zotero-cli either way);
 * the doctor reports it.
 */
export async function prepareWorkspace(spawner: Spawner, dir: string, opts: WorkspaceOpts = {}): Promise<string> {
  const env = await spawner.baseEnv();
  const made = await spawner.run("/bin/sh", ["-c", 'mkdir -p "$1"', "sh", dir], { env, timeoutMs: 10_000 });
  if (made.code !== 0) throw new Error(`could not create the workspace ${dir}: ${(made.stderr || made.stdout).trim()}`);
  for (const c of await installerCommands(spawner, env, opts)) {
    try {
      const r = await spawner.run(c.command, [...c.args, "install-skill", "--target", "claude", "--target", "agents", "--root", dir, ...(opts.refresh ? ["--force"] : [])], { cwd: dir, env, timeoutMs: 120_000 });
      // Done on success, or when the user's edited copy was left alone (exit 1, "exists and differs").
      if (r.code === 0 || /exists and differs|differs from the packaged/.test(`${r.stdout}\n${r.stderr}`)) break;
    } catch {
      // try the next way to run it
    }
  }
  return dir;
}
