## Session: 2026-02-15 18:02

### Completed
- Ran preflight diagnostics on new machine — all tools present, clean working tree
- Diagnosed MCP connection error (`[WinError 10061]`) in `k9-sniffs-claude` project
- Root cause: Zotero desktop app was not running; `ZOTERO_LOCAL=true` requires it on `localhost:23119`
- Confirmed fix: opened Zotero, verified API responding via `curl`
- No code or config changes needed

### Key Decisions
- No changes to `.mcp.json` or zotero-mcp source — config was correct, just needed Zotero running

### Next Steps
- Ensure Zotero is launched before starting Claude Code sessions that use the zotero MCP server
- Consider adding a startup check or better error message in the MCP server for when Zotero isn't running

### Open Questions
- None

## Session: 2026-08-13 (project maintenance)

### Completed
- Reviewed all 9 open PRs with local test runs against a clean baseline (1632 passing; known tiktoken/network failures excluded as environmental)
- Merged #436 (zotero_set_item_parent), #445 (lazy server import + zotero_mcp.identifiers), #444 (embeddings provider package, verified behavior-preserving), #440 (parallel extraction + transient fulltext cache, resolves #390)
- Posted verified review blockers on #442 (rate-limit guard is dead code under pyzotero 1.13.5; 4 of its own tests fail) and #443 (comma-split mangles single identifiers; dead arXiv-via-CrossRef OA fallback; CrossRef 400 collapses the batch)
- Re-verified #417 on current main (clean merge, 1674 passing, blocker resolved); left the second-search-implementation architecture call to the maintainer
- Closed #397 (umbrella superseded by agreed split; parts 1+2 landed via #440), #431 (reporter confirmed fixed on 0.9.1), #390 (auto-closed by #440)
- Fixed on this branch: #441 (web-API dedup via title-confirmed second search), #446 (attachment-less items no longer marked "PDF extraction previously failed"; in-run failure reporting), plus the two #440 follow-ups promised at merge (batch-path cache eviction + purge_stale wiring; BrokenProcessPool containment via isolated single-worker retries)
- Full suite green after each fix (1747 passing at end); changelog updated

### Key Decisions
- #442/#443 blocked rather than fixed in-place: author is active, findings are precise, and the branches live in their fork
- #417 merge deliberately left to the human maintainer (permanent second implementation of search = maintenance-commitment call)
- #401 left untouched: draft with unaddressed review feedback, author already nudged on Aug 5

### Next Steps
- Merge claude/project-maintenance-7vgy7g into main (4 fix commits + changelog)
- Untriaged issues worth a look next: #447/#448 (fulltext/notes retrieval), #428 (stale chroma fulltext after attachment deletion), #418 (collection scoping in local/hybrid), #405 (PyMuPDF in-process paths)
