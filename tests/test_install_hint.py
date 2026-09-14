"""Install hints must match how zotero-mcp was actually installed (issue #388).

A hardcoded ``pip install 'zotero-mcp-server[semantic]'`` is useless for users
who installed with ``uv tool install`` (the reporter's traceback came from
``~/.local/share/uv/tools/zotero-mcp-server/lib/python3.11/site-packages/``) or
with pipx: pip either is not on PATH or installs into a different environment,
so the missing extra never appears. The hint is now derived from the package's
own location.
"""

from zotero_mcp import utils

UV_PATH = (
    "/Users/mivaanro/.local/share/uv/tools/zotero-mcp-server/lib/python3.11/"
    "site-packages/zotero_mcp/utils.py"
)
PIPX_PATH = (
    "/Users/mivaanro/.local/pipx/venvs/zotero-mcp-server/lib/python3.11/"
    "site-packages/zotero_mcp/utils.py"
)
VENV_PATH = "/Users/mivaanro/proj/.venv/lib/python3.11/site-packages/zotero_mcp/utils.py"


def test_uv_tool_install_gets_the_uv_command(monkeypatch):
    monkeypatch.setattr(utils, "__file__", UV_PATH)

    assert utils.detect_install_flavor() == "uv"
    hint = utils.install_hint("semantic")
    assert "uv tool install --upgrade 'zotero-mcp-server[semantic]'" in hint
    # The command that silently does nothing for these users must not appear.
    assert "pip install" not in hint


def test_pipx_install_gets_the_pipx_command(monkeypatch):
    monkeypatch.setattr(utils, "__file__", PIPX_PATH)

    assert utils.detect_install_flavor() == "pipx"
    hint = utils.install_hint("pdf")
    assert "pipx install --force 'zotero-mcp-server[pdf]'" in hint
    assert "pip install" not in hint


def test_unknown_flavor_shows_every_working_command(monkeypatch):
    """Never leave a user with only a command that cannot work for them."""
    monkeypatch.setattr(utils, "__file__", VENV_PATH)

    assert utils.detect_install_flavor() is None
    hint = utils.install_hint("semantic")
    assert "pip install 'zotero-mcp-server[semantic]'" in hint
    assert "uv tool install --upgrade 'zotero-mcp-server[semantic]'" in hint
    assert "pipx install --force 'zotero-mcp-server[semantic]'" in hint


def test_install_command_without_extra_and_with_explicit_flavor():
    assert utils.install_command(flavor="pip") == "pip install 'zotero-mcp-server'"
    assert (
        utils.install_command("all", flavor="uv")
        == "uv tool install --upgrade 'zotero-mcp-server[all]'"
    )


def test_chromadb_import_error_uses_the_detected_command(monkeypatch):
    """The message in issue #388 comes from chroma_client's import guard."""
    monkeypatch.setattr(utils, "__file__", UV_PATH)

    message = f"chromadb is required for semantic search. {utils.install_hint('semantic')}"
    assert "uv tool install --upgrade 'zotero-mcp-server[semantic]'" in message
    assert "pip install" not in message


def _outside_default_dirs(monkeypatch, tmp_path, marker):
    """An environment outside the default uv/pipx directories, carrying the
    installer's own marker at its root (sys.prefix)."""
    env = tmp_path / "custom-tools" / "zotero-mcp-server"
    env.mkdir(parents=True)
    if marker:
        (env / marker).write_text("")
    monkeypatch.setattr(utils, "__file__", str(env / "lib/python3.13/site-packages/zotero_mcp/utils.py"))
    monkeypatch.setattr(utils.sys, "prefix", str(env))


def test_uv_tool_outside_the_default_directory_is_still_uv(monkeypatch, tmp_path):
    """UV_TOOL_DIR or a relocated data directory moves the environment out of
    .../uv/tools/, but uv still writes uv-receipt.toml at its root (#534)."""
    _outside_default_dirs(monkeypatch, tmp_path, "uv-receipt.toml")

    assert utils.detect_install_flavor() == "uv"
    hint = utils.install_hint("semantic")
    assert "uv tool install --upgrade 'zotero-mcp-server[semantic]'" in hint
    assert "pip install" not in hint


def test_pipx_outside_the_default_directory_is_still_pipx(monkeypatch, tmp_path):
    _outside_default_dirs(monkeypatch, tmp_path, "pipx_metadata.json")

    assert utils.detect_install_flavor() == "pipx"


def test_no_marker_outside_the_default_directories_stays_unknown(monkeypatch, tmp_path):
    _outside_default_dirs(monkeypatch, tmp_path, None)

    assert utils.detect_install_flavor() is None
