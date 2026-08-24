"""Tests for server lifespan — verifies startup is non-blocking."""

import asyncio
import json
import sys
import threading
from unittest.mock import patch

import pytest

from zotero_mcp._app import server_lifespan


@pytest.mark.asyncio
async def test_lifespan_yields_before_sync_update_completes():
    """The lifespan must yield (allow request handling) while the
    background semantic update is still running."""
    entered = threading.Event()
    proceed = threading.Event()

    def slow_update():
        entered.set()
        proceed.wait(timeout=5)

    with patch("zotero_mcp._app._sync_semantic_update", slow_update):
        async with server_lifespan(None) as ctx:
            assert ctx == {}
            # Yield to the event loop so the background task can start.
            await asyncio.sleep(0.1)
            assert entered.is_set(), \
                "_sync_semantic_update was never called in the background"
        proceed.set()


@pytest.mark.asyncio
async def test_lifespan_yields_when_update_raises():
    """Exceptions in the background update must not prevent the server
    from starting."""

    def exploding_update():
        raise RuntimeError("ChromaDB exploded")

    with patch("zotero_mcp._app._sync_semantic_update", exploding_update):
        async with server_lifespan(None) as ctx:
            assert ctx == {}
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_lifespan_yields_when_config_missing():
    """When no config exists, the background task completes instantly
    and the lifespan still yields normally."""

    def noop_update():
        pass

    with patch("zotero_mcp._app._sync_semantic_update", noop_update):
        async with server_lifespan(None) as ctx:
            assert ctx == {}


# ---------------------------------------------------------------------------
# #485: the auto-update check must not drag ChromaDB in on every startup
# ---------------------------------------------------------------------------

# Runs in a fresh interpreter because the property under test is "was this
# module ever imported", which any earlier test in the same process would have
# already decided.
_IMPORT_PROBE = """
import json, pathlib, sys

home = pathlib.Path(sys.argv[1])
(home / ".config" / "zotero-mcp").mkdir(parents=True, exist_ok=True)
if sys.argv[2] != "__none__":
    (home / ".config" / "zotero-mcp" / "config.json").write_text(sys.argv[2])

# Patch Path.home() rather than $HOME: overriding the environment variable
# also relocates the user site-packages directory, which breaks imports that
# have nothing to do with this test.
pathlib.Path.home = staticmethod(lambda: home)

from zotero_mcp._app import _sync_semantic_update

_sync_semantic_update()

print(json.dumps({
    "semantic_search": "zotero_mcp.semantic_search" in sys.modules,
    "chromadb": "chromadb" in sys.modules,
}))
"""


def _probe(tmp_path, raw_config):
    """Run ``_sync_semantic_update`` in a subprocess; report what it imported.

    ``raw_config`` is the literal bytes of ``config.json``, or None for no file
    at all — literal rather than a dict so an unparseable config is testable.
    """
    import subprocess

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(home),
         "__none__" if raw_config is None else raw_config],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "label,raw_config",
    [
        ("no config file", None),
        ("update_config absent", '{"semantic_search": {}}'),
        (
            "auto_update false",
            '{"semantic_search": {"update_config":'
            ' {"auto_update": false, "update_frequency": "startup"}}}',
        ),
        (
            "auto_update true but frequency manual",
            '{"semantic_search": {"update_config":'
            ' {"auto_update": true, "update_frequency": "manual"}}}',
        ),
        (
            "auto_update true, daily, not yet due",
            '{"semantic_search": {"update_config": {"auto_update": true,'
            ' "update_frequency": "daily", "last_update": "2126-01-01T00:00:00"}}}',
        ),
        ("unparseable config", "{not json"),
    ],
)
def test_no_due_update_never_imports_chromadb(tmp_path, label, raw_config):
    """No update due means no ChromaDB import (#485).

    The import used to sit at the top of ``_sync_semantic_update``, above the
    config check it was documented as being guarded by, so every server start
    paid for chromadb -> numpy. On Windows that import runs in the lifespan's
    worker thread and stalls every ``tools/call`` until the client times out.
    """
    seen = _probe(tmp_path, raw_config)

    assert not seen["semantic_search"], f"{label}: imported zotero_mcp.semantic_search"
    assert not seen["chromadb"], f"{label}: imported chromadb"


def test_due_update_does_import_the_heavy_module(tmp_path):
    """The guard must not become "never update" — a due run still loads it.

    ``create_semantic_search`` is stubbed out, so this asserts that the import
    happens, not that indexing succeeds.
    """
    import subprocess

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    cfg_dir = home / ".config" / "zotero-mcp"
    cfg_dir.mkdir(parents=True)
    cfg_dir.joinpath("config.json").write_text(
        '{"semantic_search": {"update_config":'
        ' {"auto_update": true, "update_frequency": "startup"}}}'
    )
    script = (
        "import json, pathlib, sys\n"
        "pathlib.Path.home = staticmethod(lambda: pathlib.Path(sys.argv[1]))\n"
        "import zotero_mcp.semantic_search as ss\n"
        "def boom(*a, **k):\n"
        "    raise RuntimeError('stop')\n"
        "ss.create_semantic_search = boom\n"
        "from zotero_mcp._app import _sync_semantic_update\n"
        "try:\n"
        "    _sync_semantic_update()\n"
        "except RuntimeError as e:\n"
        "    print(json.dumps({'reached_create': str(e) == 'stop'}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(home)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])["reached_create"] is True


def test_due_update_check_is_config_only():
    """``should_update`` must stay importable without the heavy module."""
    from zotero_mcp.update_policy import should_update

    assert should_update({"auto_update": True, "update_frequency": "startup"}) is True
    assert should_update({"auto_update": False, "update_frequency": "startup"}) is False
    assert should_update({"auto_update": True, "update_frequency": "manual"}) is False


def test_update_policy_has_no_heavy_dependencies():
    """Guard the reason ``update_policy`` exists as its own module."""
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json; import zotero_mcp.update_policy; "
            "print(json.dumps(sorted(m for m in ('chromadb', 'numpy', 'torch', "
            "'zotero_mcp.semantic_search') if m in sys.modules)))",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []
