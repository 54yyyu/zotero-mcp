import { test } from "node:test";
import assert from "node:assert/strict";
import { lexBlocks, openMathAt, settledBlock } from "../../src/ui/markdown.ts";

const lastBlock = (src: string) => lexBlocks(src).filter((t) => t.type !== "space").at(-1)!;
const shown = (src: string) => settledBlock(lastBlock(src)).map((t) => t.raw).join("").trimEnd();

test("openMathAt finds the formula that has not closed yet, and nothing else", () => {
  assert.equal(openMathAt("so the gap is $$\\Delta = \\ln"), 14);
  assert.equal(openMathAt("$$\\frac{1}{n}\\sum"), 0);
  assert.equal(openMathAt("see \\[x^2 + "), 4);
  assert.equal(openMathAt("and \\(a + "), 4);
  assert.equal(openMathAt("the ratio $R = c_w"), 10);
  // closed ones, money, escapes and code are not open formulas
  assert.equal(openMathAt("closed $$x$$ and $y$ and \\[z\\] and \\(w\\)."), -1);
  assert.equal(openMathAt("costs $5 or $10 today"), -1);
  assert.equal(openMathAt("five \\$ signs \\$x"), -1);
  assert.equal(openMathAt("run `echo $HOME` and ``a $b``"), -1);
  assert.equal(openMathAt("an open code span `echo $PATH"), -1);
  assert.equal(openMathAt("a $ alone"), -1, "a $ before a space opens nothing");
  assert.equal(openMathAt("$x\n\nnext paragraph"), -1, "a blank line ends an inline $");
  assert.equal(openMathAt("after $a$ comes $$b"), 16);
});

test("settledBlock: the streaming block stops before an open formula, a math fence or a bare table header", () => {
  assert.equal(shown("Intro.\n\nThe gap is $\\Delta = \\ln(1"), "The gap is");
  assert.equal(shown("Intro.\n\n$$\\hat\\tau = \\frac{1}"), "", "a display block shows nothing until it closes");
  assert.equal(shown("Intro.\n\n$$\\hat\\tau$$"), "$$\\hat\\tau$$", "closed: shown");
  assert.equal(shown("- one\n- two with \\(x"), "- one\n- two with", "a list keeps its items");
  assert.equal(shown("```math\n\\int_0^1"), "");
  assert.equal(shown("```math\n\\int_0^1\n```"), "```math\n\\int_0^1\n```");
  assert.equal(shown("```bash\necho $HOME"), "```bash\necho $HOME", "code is code");
  assert.equal(shown("| Study | Ratio |"), "", "a header row waits for its table");
  assert.equal(shown("| Study | Ratio |\n|---|---|\n| A | 1 |"), "| Study | Ratio |\n|---|---|\n| A | 1 |");
  assert.equal(shown("Costs $5 and"), "Costs $5 and");
  assert.equal(shown("Water is H<sub>2"), "Water is H", "a formatting tag waits for its closing tag");
  assert.equal(shown("<span style=\"color: red\">a <u>b</u>"), "");
  assert.equal(shown("H<sub>2</sub>O"), "H<sub>2</sub>O");
});
