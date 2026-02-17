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
