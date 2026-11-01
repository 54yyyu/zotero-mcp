"""zotero-cli runs the tools without loading FastMCP.

Every command that touched a tool module imported ``fastmcp`` and the MCP
SDK through it, about 0.45 s of a 1.0 s ``zotero-cli get collections``,
for a registry and an exception class the command never used. These run in
a subprocess because import cost only shows on a cold interpreter.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")

_ENTRY_PROBE = """
import json, sys
sys.path.insert(0, {src!r})
sys.argv = ["zotero-cli", "get", "collections"]
from zotero_mcp import cli_standalone
try:
    cli_standalone.main()
except SystemExit:
    pass
from zotero_mcp.tools import write
from zotero_mcp._context import ToolError
print(json.dumps({{
    "loaded": sorted(m for m in ("fastmcp", "mcp") if m in sys.modules),
    "tools_imported": "zotero_mcp.tools.search" in sys.modules,
    "plain_function": not hasattr(write.batch_update, "__fastmcp__"),
    "tool_error": ToolError.__module__,
}}))
"""


def test_the_entry_point_runs_tools_without_fastmcp(tmp_path):
    env = {
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
        "ZOTERO_LOCAL": "false",
        "ZOTERO_API_KEY": "k",
        "ZOTERO_LIBRARY_ID": "1",
        # Nothing listens here: the command fails fast after loading the tools.
        "ZOTERO_API_URL": "http://127.0.0.1:9",
        # Windows cannot load its socket provider without these (WinError 10106).
        **{k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ},
    }
    proc = subprocess.run(
        [sys.executable, "-c", _ENTRY_PROBE.format(src=SRC)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen["tools_imported"]
    assert seen["loaded"] == []
    assert seen["plain_function"]
    assert seen["tool_error"] == "zotero_mcp._serverless"


def test_the_server_still_gets_fastmcp():
    from fastmcp.exceptions import ToolError as RealToolError

    from zotero_mcp import _context, _serverless
    from zotero_mcp._app import mcp

    assert not _serverless.ENABLED
    assert _serverless.enable() is False  # tools already loaded: no-op
    assert _context.ToolError is RealToolError
    assert type(mcp).__name__ == "_ZoteroMCP"


def test_stand_in_app_leaves_functions_alone():
    from zotero_mcp._serverless import App

    app = App()

    def f():
        return 1

    assert app.tool(name="x")(f) is f
    assert app.tool(f) is f
    assert app.resource("zotero://x")(f) is f
    assert app.prompt(name="p")(f) is f
