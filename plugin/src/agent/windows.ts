// Windows: npm installs a CLI as a `.cmd` shim, and Subprocess (CreateProcess) cannot run a batch file without cmd.exe.
// cmd.exe's own quoting would corrupt an argument with a space or a quote, so the shim is read and the program it
// wraps is run directly (the same `node script.js` or `tool.exe` the shim itself ends in).

/**
 * The program an npm `.cmd` shim runs, relative to the shim's folder (`node_modules\@openai\codex\bin\codex.js`), or null
 * when it does not look like one. The shim names it as `"%dp0%\<path>" %*`; the `node.exe` it probes beside itself is skipped.
 * Node's own `npm.cmd`/`npx.cmd` are not shims: they set `NPM_CLI_JS=%~dp0\<path>` and end in `"%NODE_EXE%" "%NPM_CLI_JS%" %*`.
 */
export function shimTarget(shim: string): string | null {
  for (const m of shim.matchAll(/"%dp0%\\([^"%]+)"/g)) if (!/^node(\.exe)?$/i.test(m[1]!)) return m[1]!;
  const cli = /^"%NODE_EXE%"\s+"%(\w+)%"\s+%\*/m.exec(shim)?.[1];
  return cli ? new RegExp(`SET "${cli}=%~dp0\\\\([^"%]+)"`, "i").exec(shim)?.[1] ?? null : null;
}

/** The command line for a `.cmd`/`.bat` that is not an npm shim: through cmd.exe, which is only safe while no argument has a space or a quote. */
export const cmdLine = (comspec: string, file: string, args: string[]): [string, string[]] => [comspec, ["/d", "/c", file, ...args]];
