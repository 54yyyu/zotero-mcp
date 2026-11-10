// The harness never starts a second Zotero on its profile, and stops every process on it when a run ends.
import assert from "node:assert/strict";
import { test } from "node:test";
// @ts-expect-error: a plain .mjs script, no types
import { profilePids } from "../scripts/dev-guard.mjs";

const P = "/x/plugin/.dev/profile";
const ps = [
  "  101 /Applications/Zotero.app/Contents/MacOS/zotero --profile /x/plugin/.dev/profile --new-instance -ZoteroDebugText",
  "  102 /Applications/Zotero.app/Contents/MacOS/zotero --profile /x/plugin/.dev/profile",
  "  103 /Applications/Zotero.app/Contents/MacOS/zotero --profile /x/plugin/.dev/profile-other --new-instance",
  "  104 /Applications/Zotero.app/Contents/MacOS/zotero",
  "  105 node scripts/dev.mjs --script test/zotero/budget.js",
  "  106 /Applications/Zotero.app/Contents/MacOS/zotero --profile /x/plugin/.dev dir/profile --new-instance",
].join("\n");

test("the Zotero on our profile is found by its command line; another profile, the user's Zotero and ourselves are not", () => {
  assert.deepEqual(profilePids(ps, P, 999), [101, 102]);
  assert.deepEqual(profilePids(ps, P, 101), [102], "our own pid is never in the list");
  assert.deepEqual(profilePids(ps, "/x/plugin/.dev dir/profile", 999), [106], "a path with a space");
  assert.deepEqual(profilePids("", P, 999), []);
});
