// Build the UI preview into plugin/preview/dist (a static page: open dist/index.html or serve it).
//   node scripts/preview.mjs            build once
//   node scripts/preview.mjs --watch    rebuild on change
import { build, context } from "esbuild";
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "preview", "dist");
mkdirSync(out, { recursive: true });
copyFileSync(join(root, "preview", "index.html"), join(out, "index.html"));

const opts = {
  entryPoints: [join(root, "preview", "entry.ts")],
  outfile: join(out, "bundle.js"),
  bundle: true,
  format: "iife",
  target: ["firefox140", "chrome120", "safari17"],
  legalComments: "none",
  sourcemap: "inline",
  logLevel: "info",
  define: { "process.env.NODE_ENV": '"development"' },
};
if (process.argv.includes("--watch")) {
  const ctx = await context(opts);
  await ctx.watch();
  console.log("watching; open", join(out, "index.html"));
} else {
  await build(opts);
  console.log("built", join(out, "index.html"));
}
