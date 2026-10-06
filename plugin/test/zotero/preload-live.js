// The open paper ahead of time, measured: the same questions on a real paper through a real agent, BEFORE (the brief
// that said "read its skill here first", no metadata, no file) and AFTER (tool sheet, metadata, full-text file). Spends
// tokens on the user's subscription, so it is not in scripts/test-zotero.mjs; run it by hand (DESIGN.md "Context budget"):
//   ZMC_LIVE=1 ZMC_PAPER_PDF=paper.pdf ZMC_PAPER_META=meta.json ZMC_HEAD_BRIEF=old-brief.txt ZMC_AB=after \
//     node scripts/dev.mjs --script test/zotero/preload-live.js --timeout 900
// meta.json: {title, authors[], date, archiveID, DOI, abstract}. ZMC_BACKEND (claude-code), ZMC_MODEL (a substring of the
// model's id or name; Sonnet), ZMC_EFFORT (low), ZMC_MODE (the agent's default), ZMC_RUNS (3), ZMC_QUESTIONS (|-separated),
// ZMC_CUSTOM_FOLDER=1 runs in a chat folder of the user's own (the file is then outside the agent's folder).
// The agent's zotero-cli reads this throwaway library: ZOTERO_DB_PATH points at it, read-only (writes would go to the
// real Zotero's port, where these keys do not exist).
const QUESTIONS = ["Tell me more about this paper.", "What does the minimum description length section say?"];

async function main(ctx) {
  if (!ctx.env("ZMC_LIVE")) return { skipped: "set ZMC_LIVE=1 to run (spends tokens)" };
  const { Zotero, win, IOUtils, PathUtils } = ctx;
  const ab = ctx.env("ZMC_AB") || "after";
  const backend = ctx.env("ZMC_BACKEND") || "claude-code";
  const runs = Number(ctx.env("ZMC_RUNS") || 3);
  const questions = ctx.env("ZMC_QUESTIONS") ? ctx.env("ZMC_QUESTIONS").split("|") : QUESTIONS;
  const meta = JSON.parse(await IOUtils.readUTF8(ctx.env("ZMC_PAPER_META")));
  const headBrief = ab === "before" ? await IOUtils.readUTF8(ctx.env("ZMC_HEAD_BRIEF")) : null;

  const att = await Zotero.Attachments.importFromFile({ file: Zotero.File.pathToFile(ctx.env("ZMC_PAPER_PDF")), libraryID: Zotero.Libraries.userLibraryID });
  const item = new Zotero.Item("preprint");
  item.setField("title", meta.title);
  item.setCreators(meta.authors.map((n) => { const i = n.lastIndexOf(" "); return { creatorType: "author", firstName: n.slice(0, i), lastName: n.slice(i + 1) }; }));
  for (const f of ["date", "archiveID", "DOI"]) item.setField(f, meta[f]);
  item.setField("repository", "arXiv");
  item.setField("abstractNote", meta.abstract);
  await item.saveTx();
  att.parentID = item.id;
  await att.saveTx();
  // Into the main file, so the agent's zotero-cli sees the new items without a copy of the WAL.
  await Zotero.DB.queryAsync("PRAGMA wal_checkpoint(TRUNCATE)");
  ctx.log("imported", item.key, att.key);

  await ctx.resize(1280, 800);
  await Zotero.Reader.open(att.id);
  const reader = await ctx.waitFor(() => Zotero.Reader._readers.find((r) => r.itemID === att.id && r._internalReader), "reader");
  await reader._initPromise;
  await ctx.waitFor(() => reader._internalReader._state?.primaryViewStats?.pagesCount, "pdf loaded");
  ctx.plugin.windows.get(win).show();
  const { host, bundle } = ctx.plugin.panel();
  if (ctx.env("ZMC_CUSTOM_FOLDER")) {
    const dir = PathUtils.join(PathUtils.parent(ctx.env("ZMC_DEFAULT_CHAT_FOLDER")), "custom-project");
    await IOUtils.makeDirectory(dir, { createAncestors: true, ignoreExisting: true });
    await host.setSettings({ chatFolder: dir });
  }
  await ctx.waitFor(() => host.currentContext().some((c) => c.kind === "reader"), "reader chip");
  ctx.log("reader open");
  bundle.prefetchPaper(); // what focus in the composer does; the user types for a few seconds after it
  if (ab === "after") await ctx.waitFor(async () => (await host.paperContext(host.currentContext(), Promise.resolve())).some((c) => c.id.startsWith("fulltext:")), "the full text file", 30000);

  const db = PathUtils.join(Zotero.DataDirectory.dir, "zotero.sqlite");
  const results = [];
  for (const question of questions) {
    for (let i = 0; i < runs; i++) {
      const prep = await host.prepareSession();
      ctx.log("session prepared", prep.cwd);
      const s = await host.runtime.start({
        backend, cwd: prep.cwd, brief: headBrief ?? prep.brief, auth: backend === "pi" ? "api-key" : "subscription",
        env: { ...prep.env, ZOTERO_LOCAL: "true", ZOTERO_BACKEND: "sqlite", ZOTERO_DB_PATH: db, ...(ctx.env("ZMC_PI_AGENT_DIR") ? { PI_CODING_AGENT_DIR: ctx.env("ZMC_PI_AGENT_DIR") } : {}) },
        ...(ctx.env("ZMC_MODE") ? { mode: ctx.env("ZMC_MODE") } : {}),
      });
      const want = (ctx.env("ZMC_MODEL") || (backend === "claude-code" ? "sonnet" : "")).toLowerCase();
      const model = want && s.models().find((m) => `${m.id} ${m.name}`.toLowerCase().includes(want));
      if (model && s.currentModel() !== model.id) await s.setModel(model.id);
      const effort = s.efforts().find((e) => e.id === (ctx.env("ZMC_EFFORT") || "low"));
      if (effort) await s.setEffort(effort.id).catch((e) => ctx.log("effort not set:", String(e)));
      ctx.log("models", s.models().map((m) => m.id).join(" "), "on", s.currentModel());
      const used = host.currentContext();
      const extra = ab === "after" ? await host.paperContext(used, Promise.resolve()) : [];
      const text = [host.describeContext([...used, ...extra]).text, question].filter(Boolean).join("\n\n");
      const evs = [], perms = [];
      const t0 = Date.now();
      s.on((e) => {
        evs.push({ ...e, ms: Date.now() - t0 });
        if (e.t === "permission" && e.resolved === undefined) {
          perms.push(String(e.title).slice(0, 160));
          const o = e.options.find((x) => x.kind === "allow_once") ?? e.options[0];
          s.respondPermission(e.id, o.id);
        }
      });
      await s.prompt({ text });
      const total = Date.now() - t0;
      await s.close();
      // Each tool call once, at its first event; calls started within 500 ms of each other are one model step.
      const tools = new Map();
      for (const e of evs) {
        if (e.t !== "tool") continue;
        const t = tools.get(e.id) ?? { ms: e.ms, what: "" };
        const what = e.input?.command ?? e.input?.file_path ?? e.input?.pattern ?? e.input?.path;
        t.what = String(what ?? (t.what || e.title)).slice(0, 160);
        tools.set(e.id, t);
      }
      const starts = [...tools.values()].map((x) => x.ms).sort((a, b) => a - b);
      const steps = starts.filter((ms, k) => k === 0 || ms - starts[k - 1] > 500).length;
      const lastTool = starts.at(-1) ?? -1;
      const texts = evs.filter((e) => e.t === "text");
      const answer = texts.filter((e) => e.ms > lastTool);
      const end = evs.find((e) => e.t === "turn_end");
      const r = {
        ab, backend, model: s.currentModel(), effort: s.currentEffort(), question, run: i + 1,
        toolCalls: tools.size, steps, firstTextMs: texts[0]?.ms ?? null, firstAnswerMs: answer[0]?.ms ?? null, totalMs: total,
        contextUsed: end?.usage?.contextUsed ?? null, stop: end?.stop, permissions: perms,
        tools: [...tools.values()].map((x) => `${(x.ms / 1000).toFixed(1)}s ${x.what}`),
        answer: answer.map((e) => e.delta).join("").slice(0, 400),
      };
      ctx.log(JSON.stringify(r));
      results.push(r);
    }
  }
  const med = (xs) => { const v = xs.filter((x) => x != null).sort((a, b) => a - b); return v.length ? v[Math.floor((v.length - 1) / 2)] : null; };
  const summary = questions.map((q) => {
    const rs = results.filter((r) => r.question === q);
    const of = (k) => ({ median: med(rs.map((r) => r[k])), all: rs.map((r) => r[k]) });
    return { question: q, toolCalls: of("toolCalls"), steps: of("steps"), firstAnswerMs: of("firstAnswerMs"), totalMs: of("totalMs"), contextUsed: of("contextUsed") };
  });
  return { ab, backend, summary, results };
}
