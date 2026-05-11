"""Proxy tool — fetch paper content from a URL via an institutional proxy."""

import json
from pathlib import Path

from fastmcp import Context

from zotero_mcp._app import mcp

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"


def _load_proxy_config() -> dict | None:
    """Return the proxy config dict, or None if not configured."""
    if not _DEFAULT_CONFIG_PATH.exists():
        return None
    with open(_DEFAULT_CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg.get("proxy")


@mcp.tool(
    name="zotero_fetch_paper_from_url",
    description=(
        "Fetch a paper from a URL or DOI via your institution's proxy. "
        "Requires proxy to be configured in ~/.config/zotero-mcp/config.json. "
        "Resolves DOIs automatically. Returns JSON with status, url, and content."
    ),
)
def fetch_paper_from_url(
    url: str,
    *,
    ctx: Context,
) -> str:
    try:
        from zotero_mcp.proxy import fetch_via_proxy

        cfg = _load_proxy_config()
        if not cfg:
            return json.dumps(
                {
                    "error": (
                        "Proxy is not configured. Add a 'proxy' entry to "
                        "~/.config/zotero-mcp/config.json with 'domain' and 'browser' keys."
                    )
                }
            )

        domain = cfg.get("domain") or cfg.get("proxy_domain")
        if not domain:
            return json.dumps({"error": "No proxy domain found in config."})

        browser = cfg.get("browser", "firefox")

        ctx.info(f"Fetching paper via proxy ({domain}): {url}")
        result = fetch_via_proxy(url, browser=browser, proxy_domain=domain)
        ctx.info(f"Fetched {url} → {result['url']} (status {result['status']})")
        return json.dumps(result)

    except ImportError:
        return json.dumps({"error": "browser-cookie3 is required: pip install zotero-mcp-server[proxy]"})
    except Exception as e:
        ctx.error(f"Error fetching paper: {e}")
        return json.dumps({"error": str(e)})
