"""Discover Zotero tools and search the web through a multi-server MCP client."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from fastmcp import Client


def result_json(result):
    """Keep structured output, with a fallback for text-only MCP servers."""
    if result.structured_content is not None:
        return result.structured_content
    return [block.model_dump(mode="json") for block in result.content]


async def search(query: str, fetch_url: str | None = None):
    config = json.loads(Path(__file__).with_name("mcp.json").read_text(encoding="utf-8"))
    session_id = str(uuid4())
    async with Client(config, timeout=60, init_timeout=30) as client:
        tools = await client.list_tools()
        output = {"available_tools": [tool.name for tool in tools]}
        output["search"] = result_json(
            await client.call_tool(
                "parallel_web_search",
                {"objective": query, "search_queries": [query], "session_id": session_id},
            )
        )
        if fetch_url:
            output["fetch"] = result_json(
                await client.call_tool(
                    "parallel_web_fetch",
                    {"urls": [fetch_url], "objective": query, "search_queries": [query], "session_id": session_id},
                )
            )
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Search objective and keyword query, e.g. 'attention mechanisms original paper'")
    parser.add_argument("--fetch", metavar="URL", help="Also fetch excerpts from this HTTP(S) URL")
    args = parser.parse_args()
    try:
        output = asyncio.run(search(args.query, args.fetch))
    except Exception as error:
        print(f"MCP request failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
