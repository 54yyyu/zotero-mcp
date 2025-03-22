# Zotero MCP: Your Research Library in Claude

<p align="center">
  <img src="https://www.zotero.org/static/images/svg/zotero-logo.svg" height="60" alt="Zotero Logo">
  ➕
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Claude_Anthropic_Logo.svg" height="60" alt="Claude Logo">
</p>

**Zotero MCP** seamlessly connects your [Zotero](https://www.zotero.org/) research library with [Claude](https://www.anthropic.com/claude) and other AI assistants via the [Model Context Protocol](https://modelcontextprotocol.io/introduction). Discuss papers, get summaries, analyze citations, and more!

## 🚀 Quick Install

```bash
pip install git+https://github.com/54yyyu/zotero-mcp.git
zotero-mcp setup  # Auto-configure for Claude Desktop
```

## ✨ Features

### Search Your Library
- Find papers, articles, and books by title, author, or content
- Perform complex searches with multiple criteria
- Browse collections, tags, and recent additions

### Access Your Content
- Retrieve detailed metadata for any item
- Get full text content (when available)
- Access attachments, notes, and child items

## 🖥️ Setup & Usage

### For Claude Desktop

After installation, either:

1. **Auto-configure** (recommended):
   ```bash
   zotero-mcp setup
   ```

2. **Manual configuration**:
   Add to your `claude_desktop_config.json`:
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

### Requirements

- Python 3.10+
- Zotero 7+ (for local API with full-text access)
- Claude Desktop or compatible AI assistant

### Using Zotero in Claude

1. Start Zotero desktop (make sure local API is enabled in preferences)
2. Launch Claude Desktop
3. In Claude, type: `/tools zotero`

Example prompts:
- "Search my library for papers on machine learning"
- "Find recent articles I've added about climate change"
- "Summarize the key findings from my paper on quantum computing"

## 🔧 Advanced Configuration

### Using Web API Instead of Local API

For accessing your Zotero library via the web API (useful for remote setups):

```bash
zotero-mcp setup --no-local --api-key YOUR_API_KEY --library-id YOUR_LIBRARY_ID
```

### Environment Variables

- `ZOTERO_LOCAL=true`: Use the local Zotero API (default: false)
- `ZOTERO_API_KEY`: Your Zotero API key (for web API)
- `ZOTERO_LIBRARY_ID`: Your Zotero library ID (for web API)
- `ZOTERO_LIBRARY_TYPE`: The type of library (user or group, default: user)

### Command-Line Options

```bash
# Run the server directly
zotero-mcp serve

# Specify transport method
zotero-mcp serve --transport stdio|sse

# Get help on setup options
zotero-mcp setup --help
```

## 📚 Available Tools

### Search Tools
- `zotero_search_items`: Search your library
- `zotero_advanced_search`: Perform complex searches
- `zotero_get_collections`: List collections
- `zotero_get_collection_items`: Get items in a collection
- `zotero_get_tags`: List all tags
- `zotero_get_recent`: Get recently added items

### Content Tools
- `zotero_get_item_metadata`: Get detailed metadata
- `zotero_get_item_fulltext`: Get full text content
- `zotero_get_item_children`: Get attachments and notes

## 🔍 Troubleshooting

- **No results found**: Ensure Zotero is running and the local API is enabled
- **Can't connect to library**: Check your API key and library ID if using web API
- **Full text not available**: Make sure you're using Zotero 7+ for local full-text access

## 📄 License

MIT
