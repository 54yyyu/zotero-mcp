"""Importing a submodule must not drag in the MCP server.

``zotero_mcp/__init__.py`` used to do ``from .server import mcp`` eagerly,
so any ``import zotero_mcp.<anything>`` pulled in FastMCP, the MCP SDK,
pydantic, pyzotero, bibtexparser and unidecode first. On 0.9.1 that made
``from zotero_mcp.schema import valid_fields`` -- a stdlib-only, offline
field lookup -- cost roughly 1.45 s and ~1480 modules.

These tests pin the lazy behaviour. They run each import in a *subprocess*
because import cost is only observable on a cold interpreter: by the time
the test suite is running, ``conftest`` has already imported half the
package.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")

# Packages that only the server needs. `schema` and `identifiers` are
# stdlib-only by design and must not reach any of them.
SERVER_ONLY_MODULES = ("fastmcp", "mcp", "pydantic", "pyzotero", "bibtexparser")


def _modules_after_importing(statement: str) -> set[str]:
    """Top-level module names loaded by `statement` in a fresh interpreter."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {SRC!r})\n"
        f"{statement}\n"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


@pytest.mark.parametrize(
    "statement",
    [
        "from zotero_mcp.schema import valid_fields",
        "from zotero_mcp.identifiers import normalize_doi",
    ],
)
def test_stdlib_only_submodules_do_not_import_the_server(statement: str) -> None:
    loaded = _modules_after_importing(statement)
    leaked = sorted(set(SERVER_ONLY_MODULES) & loaded)
    assert not leaked, (
        f"{statement!r} pulled in server-only dependencies {leaked}. "
        f"Something re-introduced an eager import in zotero_mcp/__init__.py "
        f"or in the submodule itself."
    )


def test_version_is_available_without_the_server() -> None:
    loaded = _modules_after_importing("import zotero_mcp; zotero_mcp.__version__")
    leaked = sorted(set(SERVER_ONLY_MODULES) & loaded)
    assert not leaked, (
        f"`import zotero_mcp` alone pulled in {leaked}; the package root "
        f"should expose __version__ without loading the server."
    )


def test_mcp_attribute_still_resolves() -> None:
    """The lazy path must be transparent: `from zotero_mcp import mcp`
    keeps working, it just happens on first access now."""
    from zotero_mcp import mcp
    from zotero_mcp.server import mcp as direct

    assert mcp is direct


def test_mcp_is_advertised_by_dir() -> None:
    import zotero_mcp

    assert "mcp" in dir(zotero_mcp)


def test_unknown_attribute_raises_attribute_error() -> None:
    import zotero_mcp

    with pytest.raises(AttributeError):
        zotero_mcp.does_not_exist
