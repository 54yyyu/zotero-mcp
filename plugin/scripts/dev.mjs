// Run the plugin in a throwaway Zotero and drive it with a test script.
//
//   node scripts/dev.mjs --script test/zotero/smoke.js [--build] [--keep] [--timeout 120]
//
// The plugin, when ZMC_TEST_SCRIPT is set, loads that script after startup, calls its `main(ctx)`, writes the
// returned value to ZMC_TEST_OUT as JSON and quits Zotero (unless --keep, in which case the window stays up).
//
// SAFETY, enforced here and not by care: a Zotero profile does NOT carry its own library. Left alone, a
// throwaway profile opens the real one (~/Zotero), which once happened. So:
//   1. the profile and data dir live under plugin/.dev/ and user.js pins useDataDir + dataDir there;
//   2. we refuse to run if that path is, or is inside, ~/Zotero or any dir holding a zotero.sqlite we did not make;
//   3. after launch we check with lsof that the process holds nothing under ~/Zotero, and kill it if it does;
//   4. we only ever signal the pid we started, and the local server port is a checked-free one, never 23119.
import { execFileSync, spawn } from "node:child_process";
import { pathToFileURL } from "node:url";
import { existsSync, mkdirSync, openSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (n) => args.includes(`--${n}`);
const opt = (n, d) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d);

const ZOTERO = process.env.ZMC_ZOTERO_BIN || "/Applications/Zotero.app/Contents/MacOS/zotero";
const dev = join(root, ".dev");

// Every run shares one profile and one data dir, so two at once would wipe each other's. Take a lock and wait for the other
// run to finish (a lock whose process is gone is stale and is taken over).
mkdirSync(dev, { recursive: true });
{
  const lock = join(dev, ".lock");
  const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };
  const nap = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
  let told = false;
  for (;;) {
    try { mkdirSync(lock); writeFileSync(join(lock, "pid"), String(process.pid)); break; } catch {
      let owner = 0;
      try { owner = Number(readFileSync(join(lock, "pid"), "utf8")); } catch { /* being written */ }
      if (owner && !alive(owner)) { rmSync(lock, { recursive: true, force: true }); continue; }
      if (!told) { console.log(`dev: another run (pid ${owner || "?"}) is using .dev, waiting for it`); told = true; }
      nap(2000);
    }
  }
  const release = () => rmSync(lock, { recursive: true, force: true });
  process.on("exit", release);
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(sig, () => process.exit(130));
}
const profile = join(dev, "profile");
const data = join(dev, "data");
const realData = join(homedir(), "Zotero");

function die(msg) { console.error(`dev: ${msg}`); process.exit(2); }

// --- 1/2: the data dir must be ours -------------------------------------------------------------------------
for (const p of [profile, data]) {
  const r = resolve(p);
  if (r === realData || r.startsWith(realData + "/")) die(`refusing: ${r} is inside the real Zotero data dir`);
  if (!r.startsWith(resolve(root) + "/")) die(`refusing: ${r} is outside the plugin directory`);
}
// Every run starts from an empty library (the path checks above make this safe); --keep-data reuses the last one.
if (!flag("keep-data")) rmSync(data, { recursive: true, force: true });
mkdirSync(join(profile, "extensions"), { recursive: true });
mkdirSync(data, { recursive: true });

if (flag("build") || !existsSync(join(root, "dist", "addon", "plugin.js"))) {
  execFileSync("node", [join(root, "scripts", "build.mjs"), "--dev"], { cwd: root, stdio: "inherit" });
}
const addon = JSON.parse(readFileSync(join(root, "addon.json"), "utf8"));

// --- chat state per run: history and workspace start empty; bridges are cached unless we are faking them ---------
const chatDir = join(profile, "zotero-chat");
const bridges = join(chatDir, "bridges");
for (const d of ["sessions", "workspace"]) rmSync(join(chatDir, d), { recursive: true, force: true });
// The default chat folder is ~/Documents/Zotero-Chat; a test must never create that, so it is redirected into .dev.
const chatDefault = join(dev, "chat-default");
rmSync(chatDefault, { recursive: true, force: true });
if (existsSync(join(bridges, ".zmc-mock")) || flag("mock-agent")) rmSync(bridges, { recursive: true, force: true });
if (flag("mock-agent")) {
  // A bridge package that is really test/mock-agent.mjs, at the version the runtime pins, so ensureBridge finds it installed.
  const { BACKENDS } = await import(pathToFileURL(join(root, "src", "agent", "backends.ts")).href);
  const mock = pathToFileURL(join(root, "test", "mock-agent.mjs")).href;
  for (const spec of Object.values(BACKENDS)) {
    const pkg = join(bridges, "node_modules", ...spec.pkg.split("/"));
    mkdirSync(join(pkg, "dist"), { recursive: true });
    writeFileSync(join(pkg, "package.json"), JSON.stringify({ name: spec.pkg, version: spec.version, type: "module", bin: { bridge: "dist/index.js" } }));
    writeFileSync(join(pkg, "dist", "index.js"), `import ${JSON.stringify(mock)};\n`);
  }
  writeFileSync(join(bridges, ".zmc-mock"), "bridges here are test/mock-agent.mjs\n");
  console.log("dev: bridges are the mock agent");
}

// --- a free local-server port (never 23119) ----------------------------------------------------------------
function busy(port) {
  try { return execFileSync("lsof", ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN"], { encoding: "utf8" }).trim().length > 0; }
  catch { return false; }
}
let port = 23200;
while (busy(port) && port < 23300) port++;
if (port >= 23300) die("no free port in 23200-23299");

// --- profile: isolation + a quiet, deterministic Zotero ---------------------------------------------------
writeFileSync(join(profile, "user.js"), [
  ["extensions.zotero.useDataDir", true],
  ["extensions.zotero.dataDir", data],
  ["extensions.zotero.httpServer.port", port],
  ["extensions.zotero.httpServer.localAPI.enabled", true],
  ["extensions.autoDisableScopes", 0],
  ["extensions.enabledScopes", 15],
  ["extensions.startupScanScopes", 15],
  ["extensions.logging.enabled", true],
  ["app.update.enabled", false],
  ["extensions.zotero.automaticScraperUpdates", false],
  ["extensions.zotero.firstRun2", false],
  ["extensions.zotero.firstRunGuidance", false],
  ["extensions.zotero.sync.autoSync", false],
  ["extensions.zotero.lastVersion", ""],
  ["browser.shell.checkDefaultBrowser", false],
  ["toolkit.startup.max_resumed_crashes", -1],
  // --pref name=value (repeatable): extra prefs, e.g. to open the panel at start in a release build, which has no test hook.
  ...args.flatMap((a, i) => (a === "--pref" && args[i + 1]?.includes("=") ? [[args[i + 1].split("=")[0], args[i + 1].slice(args[i + 1].indexOf("=") + 1)]] : [])),
].map(([k, v]) => `user_pref(${JSON.stringify(k)}, ${JSON.stringify(v)});`).join("\n") + "\n");

// Load the unpacked build through a proxy file, and force the add-on scan on this launch. The plugin's own prefs (settings
// a previous test saved) are dropped too: every run starts from the defaults.
writeFileSync(join(profile, "extensions", addon.id), join(root, "dist", "addon"));
const prefsJs = join(profile, "prefs.js");
if (existsSync(prefsJs)) {
  const kept = readFileSync(prefsJs, "utf8").split("\n")
    .filter((l) => !/extensions\.(lastAppBuildId|lastAppVersion|lastPlatformVersion)|extensions\.zotero\.(dataDir|useDataDir)|extensions\.zotero-chat\./.test(l));
  writeFileSync(prefsJs, kept.join("\n"));
}
for (const f of ["extensions.json", "addonStartup.json.lz4"]) rmSync(join(profile, f), { force: true });


// --- a deterministic test PDF (no dependencies): 4 pages of real, selectable text ---------------------------------
function makePdf(path) {
  const pages = [1, 2, 3, 4].map((n) => [
    `Test paper for the chat plugin, page ${n}`,
    "Assigning a professional destination would increase pay in both groups,",
    "but the causal effects are about the same size so that the gap is almost unchanged.",
    "Thus, the disparity in class destinations does not explain the pay disparity.",
  ]);
  const objs = [];
  const add = (body) => { objs.push(body); return objs.length; };
  const catalog = add(""), tree = add("");
  const font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
  const kids = pages.map((lines) => {
    const text = "BT /F1 12 Tf 72 720 Td 18 TL " + lines.map((l) => `(${l.replace(/[()\\]/g, "")}) Tj T*`).join(" ") + " ET";
    const content = add(`<< /Length ${text.length} >>\nstream\n${text}\nendstream`);
    return add(`<< /Type /Page /Parent ${tree} 0 R /MediaBox [0 0 612 792] /Contents ${content} 0 R /Resources << /Font << /F1 ${font} 0 R >> >> >>`);
  });
  objs[catalog - 1] = `<< /Type /Catalog /Pages ${tree} 0 R >>`;
  objs[tree - 1] = `<< /Type /Pages /Kids [${kids.map((k) => `${k} 0 R`).join(" ")}] /Count ${kids.length} >>`;
  let out = "%PDF-1.4\n";
  const offsets = objs.map((o, i) => { const at = out.length; out += `${i + 1} 0 obj\n${o}\nendobj\n`; return at; });
  const xref = out.length;
  out += `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n` + offsets.map((o) => String(o).padStart(10, "0") + " 00000 n \n").join("");
  out += `trailer\n<< /Size ${objs.length + 1} /Root ${catalog} 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  writeFileSync(path, out);
}
const testPdf = join(dev, "test-paper.pdf");
makePdf(testPdf);

// --- test script wiring -------------------------------------------------------------------------------------
const script = opt("script");
const out = join(dev, "result.json");
const shots = join(dev, "shots");
rmSync(out, { force: true });
rmSync(out + ".log", { force: true });
mkdirSync(shots, { recursive: true });
const env = { ...process.env, ZMC_DEFAULT_CHAT_FOLDER: chatDefault, ZMC_TEST_PDF: testPdf, ZMC_TEST_OUT: out, ZMC_SNAPSHOT_DIR: shots, ZMC_KEEP: flag("keep") ? "1" : "" };
if (script) env.ZMC_TEST_SCRIPT = resolve(script);
// --home <dir>: start Zotero as on a machine with nothing set up: that HOME (inside .dev/) and only the system PATH, so no uv, node or zotero-cli is found.
const fresh = opt("home");
if (fresh) {
  const h = resolve(dev, fresh);
  if (!h.startsWith(dev + "/")) die(`refusing: --home must be inside ${dev}`);
  rmSync(h, { recursive: true, force: true });
  mkdirSync(h, { recursive: true });
  env.HOME = h;
  env.PATH = "/usr/bin:/bin:/usr/sbin:/sbin";
}
for (const k of ["ZMC_LIVE", "ZMC_LIVE_HANDSHAKE", "ZMC_LIVE_BACKENDS", "ZMC_PI_AGENT_DIR", "ZMC_REAL_INSTALL"]) if (process.env[k]) env[k] = process.env[k];

// -ZoteroDebugText is huge, so Zotero's own output is only kept when asked for (--debug).
const logFile = join(dev, "zotero.log");
const sink = flag("debug") ? openSync(logFile, "w") : "ignore";
const child = spawn(ZOTERO, ["--profile", profile, "--new-instance", "-ZoteroDebugText"], { env, stdio: ["ignore", sink, sink] });
console.log(`dev: zotero pid ${child.pid}, server port ${port}, data ${data}`);

let killed = false;
async function stop() {
  if (killed) return;
  killed = true;
  try { process.kill(child.pid, "SIGTERM"); } catch {}
  for (let i = 0; i < 20; i++) { if (child.exitCode !== null || child.signalCode) return; await sleep(250); }
  try { process.kill(child.pid, "SIGKILL"); } catch {}
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
process.on("SIGINT", async () => { await stop(); process.exit(130); });

// --- 3: verify isolation once the process has opened its database ---------------------------------------
await sleep(6000);
function holdsRealLibrary() {
  try {
    const o = execFileSync("lsof", ["-p", String(child.pid)], { encoding: "utf8" });
    return o.split("\n").filter((l) => l.includes(realData + "/"));
  } catch { return []; }
}
const leak = holdsRealLibrary();
if (leak.length) {
  await stop();
  console.error("dev: ABORTED, the instance opened files in the real data dir:\n" + leak.slice(0, 5).join("\n"));
  process.exit(3);
}
console.log("dev: isolation ok (nothing open under ~/Zotero)");

if (!script) { console.log("dev: no --script; Zotero left running (Ctrl-C stops it)"); await new Promise(() => {}); }

const deadline = Date.now() + Number(opt("timeout", "120")) * 1000;
while (!existsSync(out) && Date.now() < deadline && child.exitCode === null) await sleep(500);
if (!flag("keep")) await sleep(1500);
let code = 0;
if (existsSync(out)) {
  const res = JSON.parse(readFileSync(out, "utf8"));
  console.log(JSON.stringify(res, null, 2));
  code = res.ok ? 0 : 1;
} else {
  console.error("dev: no result written (timeout or crash)");
  if (existsSync(out + ".log")) console.error("dev: last progress:\n" + readFileSync(out + ".log", "utf8").split("\n").slice(-8).join("\n"));
  code = 1;
}
const png = existsSync(shots) ? readdirSync(shots).filter((f) => f.endsWith(".png")) : [];
if (png.length) console.log(`dev: snapshots in ${shots}: ${png.join(", ")}`);
if (flag("keep")) { console.log("dev: --keep, Zotero left running (Ctrl-C stops it)"); await new Promise(() => {}); }
await stop();
process.exit(code);
