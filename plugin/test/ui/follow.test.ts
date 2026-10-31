import { test } from "node:test";
import assert from "node:assert/strict";
import { AT_BOTTOM_PX, START, intentUp, isUpKey, pinned, scrolled } from "../../src/ui/follow.ts";
import type { Follow, Metrics } from "../../src/ui/follow.ts";

const CLIENT = 500;
const m = (scrollTop: number, scrollHeight = 5000): Metrics => ({ scrollTop, scrollHeight, clientHeight: CLIENT });

test("a trackpad scrolling up from the bottom while tokens stream lets go at once and is never pinned back", () => {
  let height = 5000;
  let top = height - CLIENT;
  let f: Follow = pinned(m(top, height));
  const tops: number[] = [];
  for (let i = 0; i < 30; i++) {
    f = intentUp(f, m(top, height));
    top -= 2;
    f = scrolled(f, m(top, height));
    height += 20; // a token arrived; the feed pins only while following
    if (f.on) { top = height - CLIENT; f = pinned(m(top, height)); }
    tops.push(top);
  }
  assert.equal(f.on, false);
  for (let i = 1; i < tops.length; i++) assert.ok(tops[i]! < tops[i - 1]!);
});

test("an upward scroll without a wheel (scrollbar drag) lets go even next to the bottom", () => {
  assert.equal(scrolled(pinned(m(4500)), m(4499)).on, false);
});

test("content shrinking clamps scrollTop up without letting go", () => {
  assert.equal(scrolled(pinned(m(4500, 5000)), m(4400, 4900)).on, true);
});

test("only a downward scroll reaching the bottom takes hold again", () => {
  let f = scrolled(pinned(m(4500)), m(4300));
  f = scrolled(f, m(4400));
  assert.equal(f.on, false);
  assert.equal(scrolled(f, m(4500 - AT_BOTTOM_PX)).on, true);
  assert.equal(scrolled(scrolled(pinned(m(4500)), m(4499)), m(4498)).on, false);
});

test("intent at the very top lets go of nothing; the up keys", () => {
  assert.equal(intentUp(START, m(0, 400)).on, true);
  assert.ok(isUpKey({ key: "PageUp" }) && isUpKey({ key: " ", shiftKey: true }) && !isUpKey({ key: " " }));
});
