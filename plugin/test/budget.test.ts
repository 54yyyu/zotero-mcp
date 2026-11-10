// Size budgets for the shipped bundles (minified, as released).
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { statSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// plugin.js is the only script read at Zotero startup, so it is the one that must stay tiny. panel.js (UI, agent runtime, katex, marked)
// is read when the panel first opens: its size is not a concern within reason, but a blow-up would be a mistake worth catching.
// (Raised 2026-10-05 with skills and prompts: panel.js was at 650k, the xpi 203k. Raised 2026-11-09 with the UI in eight
// languages: panel.js 683k -> 1018k, the xpi 227k -> 316k; the catalogs are 31k (English) to 66k (Japanese, whose text
// esbuild writes as \u escapes) each, all bundled. plugin.js reads none of them.)
const BUDGET = { "plugin.js": 12_000, "panel.js": 1_100_000, "zotero-agent.xpi": 340_000 };

test("bundles stay within their size budget", () => {
  execFileSync("node", [join(root, "scripts", "build.mjs")], { cwd: root, stdio: "pipe" });
  for (const [file, max] of Object.entries(BUDGET)) {
    const size = statSync(join(root, "dist", file.endsWith(".xpi") ? "" : "addon", file)).size;
    assert.ok(size <= max, `${file} is ${size} bytes, budget ${max}`);
  }
});
