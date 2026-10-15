// Windows: npm installs a CLI as a `.cmd` shim, and Subprocess (CreateProcess) cannot run a batch file without cmd.exe.
// cmd.exe's own quoting would corrupt an argument with a space or a quote, so the shim is read and the program it
// wraps is run directly (the same `node script.js` or `tool.exe` the shim itself ends in).

/**
 * The program an npm `.cmd` shim runs, relative to the shim's folder (`node_modules\@openai\codex\bin\codex.js`), or null
 * when it does not look like one. The shim names it as `"%dp0%\<path>" %*`; the `node.exe` it probes beside itself is skipped.
 */
export function shimTarget(shim: string): string | null {
  for (const m of shim.matchAll(/"%dp0%\\([^"%]+)"/g)) if (!/^node(\.exe)?$/i.test(m[1]!)) return m[1]!;
  return null;
}

/** The command line for a `.cmd`/`.bat` that is not an npm shim: through cmd.exe, which is only safe while no argument has a space or a quote. */
export const cmdLine = (comspec: string, file: string, args: string[]): [string, string[]] => [comspec, ["/d", "/c", file, ...args]];
