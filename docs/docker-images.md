# Docker Images and GHCR Publishing

This project publishes Docker images to GitHub Container Registry (GHCR) via `.github/workflows/docker.yml`.

## Flavors

- `core`: base install with no optional extras
- `all`: full install with `[semantic,pdf,scite]`

Flavors are selected at build time through the Docker build arg `INSTALL_EXTRAS`:

- `INSTALL_EXTRAS=""` -> core
- `INSTALL_EXTRAS="all"` -> all

## Tags

The workflow publishes these tags:

- Flavor tags: `*-core`, `*-all`
- Release tags (from `v*` git tags): `vX.Y.Z`, `vX.Y`, `vX` (plus both flavor suffixes)
- Main branch tags: `latest` (plus `latest-core`, `latest-all`)
- Immutable tags: `sha-<shortsha>` (plus flavor suffixes)

Unsuffixed tags map to the `all` flavor.

## Runtime modes

The container entrypoint supports both MCP server and CLI usage.

- Default (`ZOTERO_APP=server`):
  - Runs `zotero-mcp serve --transport stdio` when no arguments are provided
  - Any explicit args are passed to `zotero-mcp`
- CLI mode (`ZOTERO_APP=cli`):
  - Runs `zotero-cli` with provided arguments
  - Defaults to `zotero-cli --help` if no args are passed

## Environment variables for containers

Container-specific runtime variables:

- `ZOTERO_APP`: `server` (default) or `cli`
- `ZOTERO_TRANSPORT`: MCP transport used when `ZOTERO_APP=server` and no explicit args are passed (`stdio` default)

Common Zotero MCP variables (same as non-container installs):

- `ZOTERO_LOCAL=true`: use local Zotero API mode
- `ZOTERO_API_KEY`, `ZOTERO_LIBRARY_ID`, `ZOTERO_LIBRARY_TYPE`: web/hybrid access
- `ZOTERO_WEBDAV_URL`, `ZOTERO_WEBDAV_USERNAME`, `ZOTERO_WEBDAV_PASSWORD`: remote attachment download support
- `ZOTERO_EMBEDDING_MODEL`: `default`, `openai`, `gemini`, or supported HF model
- `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_BASE_URL`
- `GEMINI_API_KEY`, `GEMINI_EMBEDDING_MODEL`, `GEMINI_BASE_URL`
- `ZOTERO_DB_PATH`: override path to `zotero.sqlite`

Recommended pattern:

```bash
docker run --rm --env-file .env ghcr.io/<owner>/zotero-mcp:latest
```

## Persistence (config + ChromaDB)

By default, the container stores config and semantic index under:

- `/home/app/.config/zotero-mcp/config.json`
- `/home/app/.config/zotero-mcp/chroma_db/`

To persist configuration and ChromaDB across container restarts, mount that directory:

```bash
docker run --rm \
  -v zotero-mcp-data:/home/app/.config/zotero-mcp \
  --env-file .env \
  ghcr.io/<owner>/zotero-mcp:latest
```

If you run semantic indexing (`zotero-mcp update-db` or `zotero-cli db update`) without this mount, ChromaDB is ephemeral and will be rebuilt in a new container.

## Local build examples

```bash
# Core flavor
docker build -t zotero-mcp:core --build-arg INSTALL_EXTRAS="" .

# Full flavor
docker build -t zotero-mcp:all --build-arg INSTALL_EXTRAS=all .

# Run default MCP server mode
docker run --rm zotero-mcp:all

# Run CLI mode
docker run --rm -e ZOTERO_APP=cli zotero-mcp:all search "transformer models"

# Run with persistent config + ChromaDB
docker run --rm -v zotero-mcp-data:/home/app/.config/zotero-mcp zotero-mcp:all
```
