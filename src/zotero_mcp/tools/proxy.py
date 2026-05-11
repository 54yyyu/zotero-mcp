"""Proxy tool — fetch paper content from a URL via EZProxy."""

import json

from fastmcp import Context

from zotero_mcp._app import mcp


@mcp.tool(
    name="zotero_fetch_paper_from_url",
    description=(
        "Fetch a paper from a URL or DOI via your university's EZProxy. "
        "Requires EZProxy to be configured in ~/.config/zotero-mcp/config.json. "
        "Resolves DOIs automatically. Returns JSON with url, title, content_type, and body."
    ),
)
def fetch_paper_from_url(
    url: str,
    *,
    ctx: Context,
) -> str:
    try:
        from zotero_mcp.proxy import ProxyRegistry, fetch_paper

        registry = ProxyRegistry.from_config_file()
        if not registry.proxies:
            return json.dumps(
                {
                    "error": (
                        "EZProxy is not configured. Run: "
                        "from zotero_mcp.proxy import configure_proxy; "
                        "configure_proxy('https://%h.ezproxy.myuniversity.edu/%p')"
                    )
                }
            )

        ctx.info(f"Fetching paper via EZProxy: {url}")
        result = fetch_paper(url, registry)
        # Drop body from log but return full result
        ctx.info(f"Fetched {result['content_length']} bytes — {result['content_type']}")
        return json.dumps({k: v for k, v in result.items() if k != "body"} | {"body": result["body"]})

    except ImportError:
        return json.dumps({"error": "browser-cookie3 is required: pip install zotero-mcp-server[ezproxy]"})
    except Exception as e:
        ctx.error(f"Error fetching paper: {e}")
        return json.dumps({"error": str(e)})
