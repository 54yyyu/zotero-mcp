"""``semantic_search.persist_directory`` moves the ChromaDB index (#617).

ChromaDB fails to write its HNSW files under a non-ASCII path on Windows, and
the index used to be pinned to ``~/.config/zotero-mcp/chroma_db``. The writer
and status round trip is in ``test_chroma_client_real.py``.
"""

import builtins
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import setup_helper
from zotero_mcp.config import resolve_chroma_dir


def test_resolve_chroma_dir_defaults_under_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert resolve_chroma_dir(None) == tmp_path / ".config" / "zotero-mcp" / "chroma_db"
    assert resolve_chroma_dir("~/idx") == Path("~/idx").expanduser()


def test_setup_keeps_persist_directory(monkeypatch):
    """Setup rebuilds ``semantic_search`` from its prompts, and none of them
    asks for the index path."""
    answers = iter([
        "n",  # don't keep the existing configuration
        "1",  # default embedding model
        "1",  # manual updates
        "",  # default PDF max pages
        "",  # default extraction workers
        "",  # default chunking choice
        "",  # auto-detect DB path
    ])
    monkeypatch.setattr(builtins, "input", lambda *args: next(answers))

    config, _db_path = setup_helper.setup_semantic_search(
        {"embedding_model": "default", "persist_directory": "D:/zotero-mcp/chroma_db"}
    )

    assert config["persist_directory"] == "D:/zotero-mcp/chroma_db"


@pytest.mark.parametrize(
    "cwd, path, warns",
    [
        ("work", "chroma_db", False),
        ("work", "王林澜", True),
        # A relative path counts from the working directory.
        ("王林澜", "chroma_db", True),
    ],
)
def test_windows_warns_about_a_non_ascii_index_path(monkeypatch, caplog, tmp_path, cwd, path, warns):
    chroma_client = pytest.importorskip("zotero_mcp.chroma_client")
    (tmp_path / cwd).mkdir()
    monkeypatch.chdir(tmp_path / cwd)

    class Opened(Exception):
        pass

    def stop_before_opening(**kwargs):
        raise Opened

    monkeypatch.setattr(chroma_client, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(chroma_client.chromadb, "PersistentClient", stop_before_opening)

    with caplog.at_level(logging.WARNING, logger=chroma_client.__name__), pytest.raises(Opened):
        chroma_client.ChromaClient(persist_directory=path)

    assert ("persist_directory" in caplog.text) is warns


def _run_wizard(monkeypatch, answers, existing=None):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *args: next(it))
    config, _ = setup_helper.setup_semantic_search(existing)
    return config


def test_setup_chunking_defaults_to_disabled(monkeypatch):
    config = _run_wizard(monkeypatch, ["1", "1", "", "", "", ""])
    assert config["chunking"]["enabled"] is False
    assert config["chunking"]["max_chunks_per_item"] == 20


def test_setup_chunking_can_be_enabled_with_custom_sizes(monkeypatch, capsys):
    config = _run_wizard(
        monkeypatch,
        ["1", "1", "", "", "y", "1000", "100", "60", ""],
    )
    assert config["chunking"] == {
        "enabled": True,
        "chunk_size": 1000,
        "overlap": 100,
        "max_chunks_per_item": 60,
    }
    assert "Chunking changed" in capsys.readouterr().out


def test_setup_chunking_keeps_existing_values_on_enter(monkeypatch, capsys):
    existing = {
        "embedding_model": "default",
        "chunking": {"enabled": True, "chunk_size": 800, "overlap": 50,
                     "max_chunks_per_item": 99, "custom": "kept"},
    }
    # "n" declines keeping the existing config; Enter accepts every default.
    config = _run_wizard(
        monkeypatch, ["n", "1", "1", "", "", "", "", "", "", ""], existing
    )
    assert config["chunking"]["enabled"] is True
    assert config["chunking"]["chunk_size"] == 800
    assert config["chunking"]["max_chunks_per_item"] == 99
    assert config["chunking"]["custom"] == "kept"
    assert "Chunking changed" not in capsys.readouterr().out


def test_setup_chunking_rejects_overlap_not_smaller_than_size(monkeypatch):
    config = _run_wizard(
        monkeypatch,
        ["1", "1", "", "", "y", "500", "500", "100", "", ""],
    )
    assert config["chunking"]["overlap"] == 100


def test_setup_chunking_accepts_zero_overlap(monkeypatch):
    config = _run_wizard(
        monkeypatch,
        ["1", "1", "", "", "y", "1000", "0", "60", ""],
    )
    assert config["chunking"]["overlap"] == 0


def test_setup_chunking_warns_about_embeddings_and_memory(monkeypatch, capsys):
    _run_wizard(monkeypatch, ["1", "1", "", "", "y", "1000", "100", "60", ""])
    out = capsys.readouterr().out
    assert "large library" in out
    assert "memory" in out
    assert "up to 600,000 embeddings per 10,000 items" in out
    assert "update-db --force-rebuild" in out
    assert "The Batch API ignores chunking" in out
    warning = out[out.index("Warning:"):out.index("ignores chunking")].splitlines()
    assert len(warning) <= 4


def test_setup_chunking_warning_names_the_enabled_batch_api(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    setup_helper._prompt_chunking(None, batch_enabled=True)
    out = capsys.readouterr().out
    assert "Batch API is enabled here and ignores chunking" in out
    assert "--no-batch" in out


def _openai_existing(**extra):
    cfg = {
        "embedding_model": "openai",
        "embedding_config": {"model_name": "text-embedding-3-large", "api_key": "sk-old",
                             "base_url": "https://proxy.example/v1"},
        "openai_batch": {"enabled": False},
        "update_config": {"auto_update": True, "update_frequency": "every_5", "update_days": 5},
    }
    cfg.update(extra)
    return cfg


def test_setup_enter_keeps_every_previous_value(monkeypatch):
    prompts = []
    # n = change config; then Enter at every prompt.
    it = iter(["n"] + [""] * 20)

    def fake_input(*args):
        prompts.append(args[0] if args else "")
        return next(it)

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(setup_helper.getpass, "getpass", lambda *a: "")
    config, _ = setup_helper.setup_semantic_search(_openai_existing())

    assert config["embedding_model"] == "openai"
    assert config["embedding_config"] == {
        "model_name": "text-embedding-3-large",
        "api_key": "sk-old",
        "base_url": "https://proxy.example/v1",
    }
    assert config["update_config"]["update_frequency"] == "every_5"
    assert config["update_config"]["update_days"] == 5
    assert config["openai_batch"] == {"enabled": False}


def test_setup_can_change_just_the_update_schedule(monkeypatch):
    # Enter on model, model choice, URL, batch; "1" = manual updates.
    it = iter(["n", "", "", "", "", "1"] + [""] * 10)
    monkeypatch.setattr(builtins, "input", lambda *a: next(it))
    monkeypatch.setattr(setup_helper.getpass, "getpass", lambda *a: "")
    config, _ = setup_helper.setup_semantic_search(_openai_existing())

    assert config["embedding_config"]["api_key"] == "sk-old"
    assert config["update_config"]["update_frequency"] == "manual"


def test_setup_does_not_offer_one_providers_key_to_another(monkeypatch):
    seen = []
    it = iter(["n", "3", "", "", "3"] + [""] * 10)
    monkeypatch.setattr(builtins, "input", lambda *a: next(it))

    def fake_getpass(prompt=""):
        seen.append(prompt)
        return ""

    monkeypatch.setattr(setup_helper.getpass, "getpass", fake_getpass)
    config, _ = setup_helper.setup_semantic_search(_openai_existing())

    assert config["embedding_model"] == "gemini"
    assert "api_key" not in config["embedding_config"]
    assert "Enter keeps" not in seen[0]
    assert "base_url" not in config["embedding_config"]
