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
```
