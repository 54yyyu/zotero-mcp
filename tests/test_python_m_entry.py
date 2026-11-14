"""`python -m zotero_mcp` starts the same CLI as the `zotero-mcp` script (#837)."""

import subprocess
import sys

from zotero_mcp._version import __version__


def test_python_m_zotero_mcp_runs_the_cli():
    out = subprocess.run(
        [sys.executable, "-m", "zotero_mcp", "version"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert __version__ in out.stdout
