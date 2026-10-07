// Locate or install an ACP bridge into the plugin's data dir, once.
//   npm install --prefix <bridgeDir> <pkg>@<pinned>
// then the bridge is run as `node <entry>`: no npx at chat time (slow, needs the network).
// No node:fs here (Gecko): the check for "already installed" is a tiny `node -e`.
import type { Spawner } from "../types.ts";
import type { BackendSpec } from "./backends.ts";
import { dirnameOf, findBinary, withPathFirst } from "./env.ts";

export interface BridgeLocation {
  /** The bridge's JS entry (its package.json `bin`), absolute. */
  entry: string;
  version: string;
}

const LOCATE_SCRIPT =
  'const fs=require("fs"),path=require("path");' +
  "try{const d=path.join(process.argv[1],'node_modules',process.argv[2]);" +
  "const p=JSON.parse(fs.readFileSync(path.join(d,'package.json'),'utf8'));" +
  "const b=typeof p.bin==='string'?p.bin:Object.values(p.bin||{})[0];" +
  "const e=path.join(d,b);fs.accessSync(e);" +
  "console.log(JSON.stringify({version:p.version,entry:e}))}catch(e){}";

/** The installed bridge for `spec`, or null when it is missing (or a different version than pinned). */
export async function locateBridge(spawner: Spawner, env: Record<string, string>, node: string, bridgeDir: string, spec: BackendSpec): Promise<BridgeLocation | null> {
  const r = await spawner.run(node, ["-e", LOCATE_SCRIPT, bridgeDir, spec.pkg], { env, timeoutMs: 15_000 });
  const line = r.stdout.trim().split("\n").pop();
  if (!line) return null;
  const found = JSON.parse(line) as BridgeLocation;
  return found.version === spec.version ? found : null;
}

const inflight = new Map<string, Promise<BridgeLocation>>();

interface EnsureBridgeOpts {
  spawner: Spawner;
  /** The login-shell environment (Spawner.baseEnv()). */
  env: Record<string, string>;
  node: string;
  bridgeDir: string;
  spec: BackendSpec;
  onProgress?: (message: string) => void;
}

/** Make sure the pinned bridge is installed; a no-op (one `node -e`) when it is. Concurrent callers share one install. */
export function ensureBridge(opts: EnsureBridgeOpts): Promise<BridgeLocation> {
  const key = `${opts.bridgeDir}\u0000${opts.spec.pkg}@${opts.spec.version}`;
  let p = inflight.get(key);
  if (!p) {
    p = install(opts).finally(() => inflight.delete(key));
    inflight.set(key, p);
  }
  return p;
}

async function install(opts: EnsureBridgeOpts): Promise<BridgeLocation> {
  const { spawner, node, bridgeDir, spec } = opts;
  const env = withPathFirst(opts.env, dirnameOf(node));
  const have = await locateBridge(spawner, env, node, bridgeDir, spec);
  if (have) return have;

  const npm = await findBinary(spawner, env, "npm");
  if (!npm) throw new Error(`npm was not found on your PATH, so ${spec.label}'s bridge (${spec.pkg}) cannot be installed. Install Node.js (it includes npm).`);

  opts.onProgress?.(`Installing ${spec.pkg}@${spec.version}…`);
  const r = await spawner.run(npm, ["install", "--prefix", bridgeDir, `${spec.pkg}@${spec.version}`, "--no-audit", "--no-fund", "--loglevel=error"], {
    env,
    timeoutMs: 600_000,
  });
  if (r.code !== 0) {
    const tail = (r.stderr || r.stdout).trim().split("\n").slice(-6).join("\n");
    throw new Error(`npm could not install ${spec.pkg}@${spec.version} (exit ${r.code}).${tail ? "\n" + tail : ""}`);
  }
  const done = await locateBridge(spawner, env, node, bridgeDir, spec);
  if (!done) throw new Error(`${spec.pkg}@${spec.version} installed, but its entry point was not found in ${bridgeDir}`);
  return done;
}
