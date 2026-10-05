// Bundle src/ into dist/ and zip it as the installable .xpi.
//   node scripts/build.mjs            -> dist/<slug>.xpi
//   node scripts/build.mjs --dev      -> unminified, readable stack traces (the dev harness uses this)
import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dev = process.argv.includes("--dev");
const addon = JSON.parse(readFileSync(join(root, "addon.json"), "utf8"));
const out = join(root, "dist", "addon");

rmSync(join(root, "dist"), { recursive: true, force: true });
mkdirSync(out, { recursive: true });

// Zotero 10 refuses a manifest without applications.zotero.update_url.
writeFileSync(join(out, "manifest.json"), JSON.stringify({
  manifest_version: 2,
  name: addon.name,
  version: addon.version,
  description: addon.description,
  homepage_url: addon.homepage,
  icons: { 48: "assets/logo.svg", 96: "assets/logo.svg" }, // one SVG that follows the light/dark scheme: the same mark as the panel
  applications: { zotero: { id: addon.id, update_url: addon.updateUrl, strict_min_version: addon.minZotero, strict_max_version: "*" } },
}, null, 2));

for (const f of ["bootstrap.js", "prefpane.xhtml", "prefpane.js"]) cpSync(join(root, f), join(out, f));
cpSync(join(root, "assets"), join(out, "assets"), { recursive: true });

// Two scripts, IIFE (Gecko's subscript loader takes no module syntax): plugin.js is what Zotero loads at startup,
// panel.js (host, agent runtime, UI) is read the first time the panel opens.
const common = { bundle: true, format: "iife", target: "firefox140", platform: "neutral", mainFields: ["module", "main"], legalComments: "none", minify: !dev || !!process.env.ZMC_MINIFY, define: { "process.env.NODE_ENV": '"production"', __TEST_HOOKS__: String(dev) } };
await build({ ...common, entryPoints: [join(root, "src", "zotero", "plugin.ts")], outfile: join(out, "plugin.js"), globalName: "ZoteroChat" });
await build({ ...common, entryPoints: [join(root, "src", "zotero", "panel.ts")], outfile: join(out, "panel.js"), globalName: "ZoteroChatPanel" });

const xpi = join(root, "dist", `${addon.slug}.xpi`);
execFileSync("zip", ["-q", "-r", xpi, "."], { cwd: out });
console.log(`built ${xpi}`);
