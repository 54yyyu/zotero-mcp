"""`python -m zotero_mcp ...` is the same as `zotero-mcp ...`.

The pip-generated `zotero-mcp.exe` launcher is a separate unsigned program, which
Windows Smart App Control can block (#837); `python -m zotero_mcp` starts only
the Python that is already installed.
"""

from zotero_mcp.cli import main

if __name__ == "__main__":
    main()
