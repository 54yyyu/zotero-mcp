// Regenerate updates.json, the manifest Zotero polls (addon.json "updateUrl") to find a newer version.
//   node scripts/release-updates.mjs <tag | xpi-url> [--xpi <file>]
//
// To release:
//   1. Tag the repo (release.yml builds the xpi and attaches it to that GitHub release).
//   2. node scripts/release-updates.mjs v0.14.0   (fetches the released xpi and hashes it)
//   3. Commit updates.json to main; installed plugins read it from there.
// Hash the file that was released, not a local rebuild: zip output is not reproducible, so a local
// build hashes differently and Zotero would refuse the update. --xpi hashes a file you downloaded
// from the release instead of fetching it. The version comes from addon.json: bump it before tagging.
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const addon = JSON.parse(readFileSync(join(root, "addon.json"), "utf8"));

const args = process.argv.slice(2);
const xpiFlag = args.indexOf("--xpi");
const file = xpiFlag >= 0 ? args.splice(xpiFlag, 2)[1] : null;
const [target] = args;
if (!target || (xpiFlag >= 0 && !file)) {
  console.error("usage: node scripts/release-updates.mjs <tag | xpi-url> [--xpi <file>]");
  process.exit(2);
}
const url = /^https?:\/\//.test(target) ? target : `${addon.homepage}/releases/download/${target}/${addon.slug}.xpi`;

let bytes;
if (file) {
  bytes = readFileSync(file);
} else {
  const res = await fetch(url);
  if (!res.ok) {
    console.error(`${url}: HTTP ${res.status}. Is the release published with ${addon.slug}.xpi attached?`);
    process.exit(1);
  }
  bytes = Buffer.from(await res.arrayBuffer());
}

const manifest = {
  addons: {
    [addon.id]: {
      updates: [{
        version: addon.version,
        update_link: url,
        update_hash: `sha256:${createHash("sha256").update(bytes).digest("hex")}`,
        applications: { zotero: { strict_min_version: addon.minZotero } },
      }],
    },
  },
};
writeFileSync(join(root, "updates.json"), JSON.stringify(manifest, null, 2) + "\n");
console.log(`updates.json: ${addon.id} ${addon.version} -> ${url}`);
