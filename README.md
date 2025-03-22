# Model Context Protocol Server for Zotero

This project is a Python server that implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) for [Zotero](https://www.zotero.org/), giving you access to your Zotero library within AI assistants. It uses the [pyzotero](https://github.com/urschrei/pyzotero) library to interact with Zotero's API, allowing AI assistants to search your Zotero library, retrieve metadata, and access full text content.

## Features

This MCP server provides the following tools:

- **Search functionality**
  - `zotero_search_items`: Search for items in your Zotero library by title, creator, or full text
  - `zotero_advanced_search`: Perform advanced searches with multiple criteria
  - `zotero_get_collections`: List all collections in your Zotero library
  - `zotero_get_collection_items`: Get all items in a specific collection
  - `zotero_get_tags`: Get all tags used in your Zotero library
  - `zotero_get_recent`: Get recently added items to your library

- **Item access**
  - `zotero_get_item_metadata`: Get detailed metadata for a specific item
  - `zotero_get_item_fulltext`: Get the full text content of a specific item
  - `zotero_get_item_children`: Get child items (attachments, notes) for a specific item

All tool responses are formatted using Markdown for optimal readability when used with AI assistants.

## Installation

### Quick Install from GitHub

The easiest way to install is directly from GitHub:

```bash
pip install git+https://github.com/54yyyu/zotero-mcp.git
```

### Using pip (once published)

```bash
pip install zotero-mcp
```

### From source

```bash
git clone https://github.com/54yyyu/zotero-mcp.git
cd zotero-mcp
pip install -e .
```

## Quick Setup for Claude Desktop

After installation, you can automatically configure zotero-mcp for Claude Desktop with a single command:

```bash
zotero-mcp setup
```

This will:
1. Find your zotero-mcp installation
2. Locate (or create) your Claude Desktop configuration file
3. Configure it to use the local Zotero API

### Advanced Setup Options

For more options:

```bash
zotero-mcp setup --help
```

Options include:
- `--no-local`: Configure for the Zotero Web API instead of the local API
- `--api-key`: Your Zotero API key (for Web API)
- `--library-id`: Your Zotero library ID (for Web API)
- `--library-type`: Your Zotero library type (user or group, for Web API)
- `--config-path`: Manually specify the Claude Desktop config file path

## Configuration

The server can be configured through environment variables:

- `ZOTERO_LOCAL=true`: Use the local Zotero API (default: false)
- `ZOTERO_API_KEY`: Your Zotero API key (not required for the local API)
- `ZOTERO_LIBRARY_ID`: Your Zotero library ID (your user ID for user libraries, not required for the local API)
- `ZOTERO_LIBRARY_TYPE`: The type of library (user or group, default: user)

### Local Zotero Instance (Zotero 7+)

If you're running Zotero 7 or newer, you can use this server with the local API. Make sure to:

1. Enable the local API in Zotero's preferences
2. Set `ZOTERO_LOCAL=true` in your environment variables or use the setup helper

> **Note:** For access to the fulltext API locally, you'll need Zotero 7 or newer. Earlier versions don't support this feature.

### Manual Claude Desktop Integration

If you prefer to manually configure Claude Desktop, add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "zotero": {
      "command": "zotero-mcp",
      "env": {
        "ZOTERO_LOCAL": "true"
      }
    }
  }
}
```

## Usage

Once configured, you can run the server directly:

```bash
zotero-mcp serve
```

Or run it with specific options:

```bash
zotero-mcp serve --transport stdio|sse
```

## Development

We welcome contributions! To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/54yyyu/zotero-mcp.git
cd zotero-mcp

# Install in development mode
pip install -e .
```

## License

MIT
