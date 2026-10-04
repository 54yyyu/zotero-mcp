// The one-click zotero-cli install on a machine with nothing (run with --home <dir> inside .dev/: that HOME and only the
// system PATH): uv is fetched, then zotero-cli, then it works. Real Gecko subprocesses, pipes and PATH lookups, but no
// network by default: the two downloads are replaced by a stub that writes a fake uv, whose `tool install` writes a fake
// zotero-cli. ZMC_REAL_INSTALL=1 downloads for real (about 30 s; it can be slow on a bad connection: use --timeout 600).
const STUB_INSTALLER = `
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/uv" <<'EOS'
#!/bin/sh
echo "stub uv $*"
printf '#!/bin/sh\\necho "usage: zotero-cli (stub)"\\n' > "$HOME/.local/bin/zotero-cli"
chmod +x "$HOME/.local/bin/zotero-cli"
EOS
chmod +x "$HOME/.local/bin/uv"
echo "stub installer ran"
`;

async function main(ctx) {
  const { bundle } = ctx.plugin.panel();
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const home = ctx.env("HOME");
  check(/\/plugin\/\.dev\//.test(home), "this must run with --home, never in a real HOME: " + home);
  const real = !!ctx.env("ZMC_REAL_INSTALL");
  if (!real) {
    const spawn = bundle.spawner.spawn.bind(bundle.spawner);
    bundle.spawner.spawn = (cmd, args, opts) => spawn(...(args.join(" ").includes("astral.sh") ? ["/bin/sh", ["-c", STUB_INSTALLER]] : [cmd, args]), opts);
  }

  const before = await bundle.host.doctor();
  const cli = before.find((c) => c.id === "cli");
  check(cli && !cli.ok && cli.fix, "a machine with nothing reports zotero-cli missing, with a fix: " + JSON.stringify(cli));
  const lines = [];
  for await (const line of cli.fix.run()) { lines.push(line); ctx.log(line.slice(0, 160)); }
  check(lines.includes("Installed."), "the install reported success: " + lines.slice(-5).join(" | "));
  check(real || lines.includes("stub uv tool install --upgrade zotero-mcp-server"), "uv, once fetched, ran the install: " + lines.join(" | "));

  const found = (await bundle.host.doctor()).find((c) => c.id === "cli");
  check(found.ok && found.detail === `${home}/.local/bin/zotero-cli`, "zotero-cli is found in ~/.local/bin, which was never on this PATH: " + JSON.stringify(found));
  const run = await bundle.spawner.run(found.detail, ["--help"], { env: await bundle.spawner.baseEnv(), timeoutMs: 60000 });
  check(run.code === 0 && /usage: zotero-cli/.test(run.stdout + run.stderr), "the installed zotero-cli runs: exit " + run.code + " " + (run.stderr || run.stdout).slice(0, 120));
  return { mode: real ? "real download" : "stub", found: found.detail, lastLines: lines.slice(-3) };
}
