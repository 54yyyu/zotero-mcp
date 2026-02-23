# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-02-20

### Changed
- Published to PyPI as `zotero-mcp-server`. Install with `pip install zotero-mcp-server`.
- Updater now checks PyPI for latest versions (with GitHub releases as fallback).
- Updater now installs/upgrades from PyPI instead of git URLs.
- Install instructions updated to use PyPI in README and docs.

### Added
- PyPI badge in README.
- `keywords`, `license`, and additional `project.urls` metadata in package config.
- This changelog.

### Fixed
- Cleaned up `MANIFEST.in` (removed reference to nonexistent `setup.py`).

## [0.1.2] - 2026-01-07

### Added
- Full-text notes integration for semantic search.
- Extra citation key display support (Better BibTeX).

## [0.1.1] - 2025-12-29

### Added
- EPUB annotation support with CFI generation.
- Annotation feature documentation.
- Semantic search with ChromaDB and multiple embedding model support (default, OpenAI, Gemini).
- Smart update system with installation method detection.
- ChatGPT integration via SSE transport and tunneling.
- Cherry Studio and Chorus client configuration support.

## [0.1.0] - 2025-03-22

### Added
- Initial release.
- Zotero local and web API integration via pyzotero.
- MCP server with stdio transport.
- Claude Desktop auto-configuration (`zotero-mcp setup`).
- Search, metadata, full-text, collections, tags, and recent items tools.
- PDF annotation extraction with Better BibTeX support.
- Smithery and Docker support.
