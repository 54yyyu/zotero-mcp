import { test } from "node:test";
import assert from "node:assert/strict";
import { stepTitle } from "../../src/ui/steptitle.ts";

test("zotero-cli calls read as what they do", () => {
  const cases: [string, string][] = [
    ["cd /tmp && zotero-cli --json read HGEXED7Q --start-page 7 --end-page 8", "Read pages 7–8 of HGEXED7Q"],
    ["zotero-cli read HGEXED7Q --start-page 3", "Read page 3 of HGEXED7Q"],
    ["zotero-cli read HGEXED7Q --start-page=3 --end-page=3 --format image", "Read page 3 of HGEXED7Q as images"],
    ["`zotero-cli --json get metadata HGEXED7Q`", "Get metadata of HGEXED7Q"],
    ["zotero-cli get children HGEXED7Q 2>&1", "Get attachments and notes of HGEXED7Q"],
    ['zotero-cli search "racial discrimination hiring" --limit 5', "Search library for “racial discrimination hiring”"],
    ["zotero-cli --json search --mode semantic 'callback gap'", "Semantic search for “callback gap”"],
    ["zotero-cli annotations list HGEXED7Q", "List annotations of HGEXED7Q"],
    ["zotero-cli notes create HGEXED7Q --text hi", "Add a note to HGEXED7Q"],
    ["zotero-cli open HGEXED7Q --page 4", "Open HGEXED7Q at page 4"],
    ["zotero-cli open --annotation ABCD1234", "Open annotation ABCD1234"],
    ["zotero-cli get collections", "List collections"],
  ];
  for (const [raw, want] of cases) assert.equal(stepTitle(raw), want, raw);
});

test("anything less clear stays the command, tidied: no cd prefix, ~ for home, skill paths relative", () => {
  assert.equal(stepTitle("cd /tmp && cat /tmp/p1.txt; head -5 /tmp/p2.txt"), "cat /tmp/p1.txt; head -5 /tmp/p2.txt");
  assert.equal(stepTitle('cd "/Users/yiyu/Documents/Zotero Chat" && ls'), "ls");
  assert.equal(stepTitle('cat "/Users/yiyu/Documents/Zotero Chat/.agents/skills/zotero-cli/SKILL.md"'), 'cat ".agents/skills/zotero-cli/SKILL.md"');
  assert.equal(stepTitle('cat "/Users/yiyu/Documents/Zotero Chat/.agents/sk...'), 'cat ".agents/sk...');
  assert.equal(stepTitle("Read /home/ana/Zotero Chat/.claude/skills/zotero-cli/SKILL.md"), "Read .claude/skills/zotero-cli/SKILL.md");
  assert.equal(stepTitle("ls /tmp /x/.agents/skills"), "ls /tmp .agents/skills");
  assert.equal(stepTitle("ls /Users/yiyu/Downloads /Users/yiyu"), "ls ~/Downloads ~");
  assert.equal(stepTitle("zotero-cli read KEY --start-page 2 > /tmp/p.txt"), "zotero-cli read KEY --start-page 2 > /tmp/p.txt");
  assert.equal(stepTitle("zotero-cli frobnicate KEY"), "zotero-cli frobnicate KEY");
  assert.equal(stepTitle("Searching the web"), "Searching the web");
  assert.equal(stepTitle(""), "");
});
