// Every in-Zotero suite, in a throwaway Zotero each (see dev.mjs for the isolation rules).
//   node scripts/test-zotero.mjs            no tokens spent
//   ZMC_LIVE=1 node scripts/test-zotero.mjs also runs the live Claude test (a few hundred tokens)
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const welcomed = ["--pref", 'extensions.zotero-chat.settings={"welcomed":true}']; // the suites that drive the chat start past the first-run welcome
const suites = [
  ["budget", []],
  ["context", []],
  ["host", []],
  ["chat", ["--mock-agent", ...welcomed]],
  ["context-economy", ["--mock-agent", ...welcomed]], // the same selection twice is sent once; real block sizes
  ["ui", ["--mock-agent", ...welcomed]],
  ["ring", ["--mock-agent", ...welcomed]], // the context ring's tooltip and popover in real Gecko: shown, inside the window, light and dark
  ["appearance", ["--mock-agent", ...welcomed]],
  ["prefpane", ["--mock-agent", ...welcomed]], // the Zotero Chat pane in Zotero's Settings: lazy, no agent started, live both ways // glass, accent, a preset and a picture through the real host; persists across a remount
  ["diagram", ["--mock-agent", ...welcomed]], // ```svg answers as themed figures; Copy and Save through the host
  ["notes", ["--mock-agent", ...welcomed]], // Save as note: real child/standalone notes, math and links in the HTML, diagrams as embedded images the editor renders
  ["translate", ["--mock-agent", ...welcomed]],
  ["skills", ["--mock-agent", ...welcomed]], // skills and prompts: lazy scan, import through the real host, sync into the chat folder, /name as a plain message // the reader's Translate: nothing loads until pressed, the answer streams into Zotero's own popup, the chat is untouched
  ["cite", ["--mock-agent"]], // a quote link flashes the sentence in the reader
  ["welcome", ["--mock-agent"]],
  ["agent", []], // the real bridge's handshake: no prompt, no tokens
  ["cli-install", ["--home", "home-fresh"]], // a machine with nothing: uv, then zotero-cli (stubbed downloads; ZMC_REAL_INSTALL=1 for real)
  ...(process.env.ZMC_LIVE ? [["live", []]] : []),
];
let failed = 0;
for (const [name, extra] of suites) {
  const r = spawnSync("node", [join(root, "scripts", "dev.mjs"), "--script", `test/zotero/${name}.js`, "--build", ...extra], { cwd: root, encoding: "utf8" });
  const ok = /"ok": true/.test(r.stdout);
  console.log(`${ok ? "ok  " : "FAIL"} ${name}`);
  if (!ok) { failed++; console.log((r.stdout + r.stderr).split("\n").filter((l) => /error|FAILED|dev:/.test(l)).slice(0, 6).join("\n")); }
}
process.exit(failed ? 1 : 0);
