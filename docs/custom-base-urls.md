# Custom Base URL Configuration Examples

This document shows how to configure custom base URLs for OpenAI and Gemini models in zotero-mcp.

## Environment Variables

Set these environment variables to use custom endpoints:

```bash
# OpenAI custom endpoint
export OPENAI_BASE_URL="https://your-openai-proxy.com/v1"

# Gemini custom endpoint  
export GEMINI_BASE_URL="https://your-gemini-proxy.com"

# Standard configuration
export ZOTERO_EMBEDDING_MODEL="openai"  # or "gemini"
export OPENAI_API_KEY="your-api-key"
# or
export GEMINI_API_KEY="your-api-key"
```

## Interactive Setup

Run the setup command and it will prompt for base URLs:

```bash
zotero-mcp setup
```

During setup, you'll be asked:
- "Enter custom OpenAI base URL (leave blank for default):"
- "Enter custom Gemini base URL (leave blank for default):"

## Claude Desktop Configuration

Your `claude_desktop_config.json` will include the base URLs:

```json
{
  "mcpServers": {
    "zotero": {
      "command": "zotero-mcp",
      "env": {
        "ZOTERO_LOCAL": "true",
        "ZOTERO_EMBEDDING_MODEL": "openai",
        "OPENAI_API_KEY": "your-api-key",
        "OPENAI_BASE_URL": "https://your-proxy.com/v1"
      }
    }
  }
}
```

## Use Cases

### Azure OpenAI
```bash
export OPENAI_BASE_URL="https://your-resource.openai.azure.com/"
export OPENAI_API_KEY="your-azure-api-key"
```

### Local/Self-hosted OpenAI Compatible API
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="local-api-key"
```

### Custom Gemini Proxy
```bash
export GEMINI_BASE_URL="https://gemini-proxy.yourcompany.com"
export GEMINI_API_KEY="your-proxy-key"
```

## Configuration File

Base URLs are saved in `~/.config/zotero-mcp/config.json`:

```json
{
  "semantic_search": {
    "embedding_model": "openai",
    "embedding_config": {
      "api_key": "your-key",
      "model_name": "text-embedding-3-small", 
      "base_url": "https://your-proxy.com/v1"
    }
  }
}
```

The configuration is automatically loaded when zotero-mcp starts.