import { test } from "node:test";
import assert from "node:assert/strict";
import { LIMIT, arrange, countLabel, providerOf, search, shortName, withHeadings } from "../../src/ui/model-list.ts";
import { CATALOGS, bigPiCatalog } from "../../src/ui/fake-catalog.ts";

const claude = CATALOGS["claude-code"].models;
const pi = bigPiCatalog();
const ids = (rows: { m: { id: string } }[]) => rows.map((r) => r.m.id);

test("providers and short names come from the id: the part before the first slash", () => {
  assert.equal(providerOf("openrouter/moonshotai/kimi-k2.6"), "openrouter");
  assert.equal(providerOf("sonnet"), "");
  assert.equal(shortName({ id: "openrouter/moonshotai/kimi-k2.6", name: "openrouter/MoonshotAI: Kimi K2.6" }), "MoonshotAI: Kimi K2.6");
  assert.equal(shortName({ id: "qwen38/qwen38-27b", name: "Qwen 27B" }), "Qwen 27B", "a name without the prefix stays as it is");
  assert.equal(shortName({ id: "opus", name: "Opus 5.5" }), "Opus 5.5");
});

test("a short catalog keeps the bridge's order: the first four, the current and the default on top, nothing pinned or grouped", () => {
  const l = arrange(claude, "sonnet", "opus");
  assert.deepEqual([l.long, l.grouped], [false, false]);
  assert.deepEqual(ids(l.rows), claude.map((m) => m.id));
  assert.deepEqual(ids(l.rows.filter((r) => r.top)), ["opus", "sonnet", "haiku", "opus-4-1"]);
  const beyond = arrange(claude, "haiku-3-5", "opus");
  assert.deepEqual(ids(beyond.rows.filter((r) => r.top)), ["opus", "sonnet", "haiku", "opus-4-1", "haiku-3-5"], "the current one is listed without expanding");
  const def = arrange(claude, "opus", "sonnet-4");
  assert.ok(def.rows.find((r) => r.m.id === "sonnet-4")!.top, "so is the default, to go back to it");
  assert.equal(def.rows[0]!.label, "Claude Opus", "labels are the names");
});

test("a long catalog: the current model, then the default, then the smaller providers' models, then the big one in its own order", () => {
  const l = arrange(pi.models, "ollama/qwen3-8b", pi.model);
  assert.deepEqual([l.long, l.grouped, l.rows.length], [true, true, 418]);
  assert.deepEqual(ids(l.rows.slice(0, 3)), ["ollama/qwen3-8b", "openrouter/moonshotai/kimi-k2.6", "my-cluster/deepseek-v4-flash"]);
  assert.deepEqual(l.rows.slice(0, 3).map((r) => r.pinned), [true, true, false]);
  assert.equal(l.rows.filter((r) => r.m.id === pi.model).length, 1, "a pinned model is not listed twice");
  const rest = l.rows.slice(3).filter((r) => r.provider === "openrouter").map((r) => r.m.id);
  assert.deepEqual(rest, pi.models.filter((m) => m.id.startsWith("openrouter/") && m.id !== pi.model).map((m) => m.id), "OpenRouter keeps the bridge's order");
  assert.equal(l.rows.some((r) => r.top), false, "no More models fold in a long list");
  // pinned rows say their provider; grouped rows drop the prefix (the heading has it)
  assert.deepEqual([l.rows[1]!.label, l.rows[1]!.note], ["MoonshotAI: Kimi K2.6", "openrouter"]);
  assert.equal(l.rows[2]!.label, "DeepSeek V4 Flash (4xH100)");
  // the current model is the default: listed once
  assert.deepEqual(ids(arrange(pi.models, pi.model, pi.model).rows.slice(0, 2)), [pi.model, "my-cluster/deepseek-v4-flash"]);
});

test("a long catalog of one provider is not grouped and keeps full names", () => {
  const one = Array.from({ length: 40 }, (_, i) => ({ id: `m-${i}`, name: `Model ${i}` }));
  const l = arrange(one, "m-30", undefined);
  assert.deepEqual([l.long, l.grouped], [true, false]);
  assert.deepEqual(ids(l.rows.slice(0, 2)), ["m-30", "m-0"]);
  assert.deepEqual(withHeadings(l.rows, l.grouped), l.rows, "no headings");
});

test("headings start each provider's group; pinned rows have none", () => {
  const l = arrange(pi.models, pi.model, pi.model);
  const items = withHeadings(l.rows.slice(0, 4), l.grouped);
  assert.deepEqual(items.map((x) => ("heading" in x ? `# ${x.heading}` : x.m.id)),
    [pi.model, "# my-cluster", "my-cluster/deepseek-v4-flash", "# ollama", "ollama/qwen3-8b", "# openrouter", "openrouter/anthropic/claude-3-haiku"]);
});

test("search: every word must match the name, id or description, in any case; order is kept; blank is everything", () => {
  const l = arrange(pi.models, pi.model, pi.model);
  const opus = search(l, "claude opus 4");
  assert.deepEqual(opus.map((r) => r.label), ["Anthropic: Claude Opus 4", "Anthropic: Claude Opus 4.1", "Anthropic: Claude Opus 4.5", "Anthropic: Claude Opus 4.6", "Anthropic: Claude Opus 4.8"]);
  assert.deepEqual(ids(search(l, "  KIMI   k2.6 ")), [pi.model], "case and spacing do not matter");
  assert.deepEqual(ids(search(l, "my-cluster")), ["my-cluster/deepseek-v4-flash"], "the id matches too");
  assert.equal(search(l, "").length, 418);
  assert.deepEqual(search(l, "opus nonsense"), []);
  const c = arrange(claude, "sonnet", "opus");
  assert.deepEqual(ids(search(c, "previous")), ["opus-4-1", "sonnet-4"], "descriptions match");
  assert.equal(countLabel(1), "1 model");
  assert.equal(countLabel(12), "12 models");
});

test("speed: arranging 418 models and 200 searches stay well under a frame", () => {
  const t0 = performance.now();
  const l = arrange(pi.models, pi.model, pi.model);
  const t1 = performance.now();
  const qs = ["c", "cl", "cla", "clau", "claude", "claude o", "claude op", "claude opu", "claude opus", "claude opus 4"];
  for (let i = 0; i < 20; i++) for (const q of qs) search(l, q).slice(0, LIMIT);
  const t2 = performance.now();
  assert.ok(t1 - t0 < 20, `arrange took ${(t1 - t0).toFixed(2)} ms`);
  assert.ok((t2 - t1) / 200 < 2, `a search took ${((t2 - t1) / 200).toFixed(3)} ms`);
});
