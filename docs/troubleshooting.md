# Troubleshooting

## General issues

- **No results found**: Ensure Zotero is running and the local API is enabled. You need to toggle on `Allow other applications on this computer to communicate with Zotero` in Zotero preferences.
- **Can't connect to library**: Check your API key and library ID if using the web API, and that the key has the permissions you need.
- **Full text not available**: Make sure you're using Zotero 7+ for local full-text access.
- **Claude Desktop can't find the `zotero-mcp` command**: use the absolute path instead (run `zotero-mcp setup-info` or `which zotero-mcp` to find it) — GUI apps don't always inherit your shell `PATH`.
- **Settings seem ignored**: environment variables set in the shell you launch a client from override the values in its config file.
- **Where to look for errors**: the Claude Desktop logs, or the MCP server's own output.
- **Installation/search option switching issues**: Database problems from changing install methods or search options can often be resolved with `zotero-mcp update-db --force-rebuild` (see [Database issues](#database-issues)).

<a id="local-library-limitations"></a>

### Local library limitations

**On Zotero 10 or newer** the local API accepts writes. Run `zotero-mcp authorize-local`
once, choose "Always Allow" in the dialog Zotero shows, and tagging, item edits, notes,
collections and file attachments all work against the local library with no cloud
account. See [Local write support](configuration.md#local-write-support).

**On Zotero 9 and older** the local API is read-only, so library modifications will not
work over the local connection alone. Set `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID`
alongside `ZOTERO_LOCAL=true` — the server then reads locally and writes through the web
API ("hybrid mode"), which is what makes tagging and the rest behave as expected.

<a id="semantic-search"></a>

## Semantic search issues

- **"Missing required environment variables" when running update-db**: Run `zotero-mcp setup` to configure your environment, or the CLI will automatically load settings from your MCP client config (e.g., Claude Desktop)
- **ChromaDB / stale embedding model errors**: If you changed embedding models and see 404 errors (e.g., `text-embedding-004 is not found`), run `zotero-mcp update-db --force-rebuild` to recreate the collection with your current model. If that doesn't work, delete `~/.config/zotero-mcp/chroma_db/` and rebuild.
- **Database update takes long**: By default, `update-db` is fast (metadata-only). For comprehensive indexing with full-text, use `--fulltext` flag. Use `--limit` parameter for testing: `zotero-mcp update-db --limit 100` (a `--limit` run does not advance the sync watermark, so a later plain `update-db` still picks up everything it skipped)
- **Semantic search returns no results**: Ensure the database is initialized with `zotero-mcp update-db` and check status with `zotero-mcp db-status`
- **Limited search quality**: For better semantic search results, use `zotero-mcp update-db --fulltext` to index full-text content (requires local Zotero setup)
- **OpenAI/Gemini API errors**: Verify your API keys are correctly set and have sufficient credits/quota
- **Ollama `Read timed out`**: see the `timeout` and `request_batch_size` settings in [Semantic search](semantic-search.md#ollama)

<a id="database-issues"></a>

### Database issues

Switching installs or install methods (sometimes to deal with failed installs), as well as toggling between search options, can sometimes lead to database problems. These can frequently be solved with:

```bash
zotero-mcp update-db --force-rebuild
```

A forced rebuild deletes the whole ChromaDB collection and re-embeds every item in the active library from scratch. With OpenAI or Gemini embeddings that is billed again in full, and it takes as long as the first build. If the index also holds other libraries (or documents with no library attribution), the command refuses and asks for `--allow-mass-deletion`; passing that flag drops those documents permanently. Back up `~/.config/zotero-mcp/chroma_db/` first, and try a plain `zotero-mcp update-db` before rebuilding.

## Update issues

- **Update command fails**: Check your internet connection and try `zotero-mcp update --force`
- **Configuration lost after update**: The update process preserves configs automatically, but check `~/.config/zotero-mcp/` for backup files

For more help, try the [discussions](https://github.com/54yyyu/zotero-mcp/discussions) or the [Discord](https://discord.gg/BvgjbcBUqg).
