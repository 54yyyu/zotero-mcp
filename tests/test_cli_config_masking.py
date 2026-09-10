"""`zotero-cli config` prints resolved settings, and the packaged skill tells
agents to run it first, so every key-shaped value must be masked by default.
The mask list used to name only Zotero/WebDAV keys; OPENAI_API_KEY and
GOOGLE_API_KEY were printed in full.
"""

import argparse
import json

from zotero_mcp import cli_standalone
from zotero_mcp.cli import obfuscate_config_for_display


def test_provider_api_keys_are_masked():
    config = {
        "ZOTERO_API_KEY": "zzzz1234secret",
        "OPENAI_API_KEY": "sk-openai-1234567890",
        "GOOGLE_API_KEY": "AIza-google-1234567890",
        "GEMINI_API_KEY": "AIza-gemini-1234567890",
        "ZOTERO_LOCAL": "true",
    }
    shown = obfuscate_config_for_display(config)

    assert shown["ZOTERO_LOCAL"] == "true"
    for key in ("ZOTERO_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        assert shown[key] != config[key], key
        assert shown[key].startswith(config[key][:4]), key
        assert set(shown[key][4:]) == {"*"}, key


def test_secret_shaped_suffixes_are_masked_generically():
    config = {
        "SOME_VENDOR_API_KEY": "abcd-efgh-ijkl",
        "SOME_VENDOR_PASSWORD": "hunter2hunter2",
        "SOME_VENDOR_TOKEN": "tok-1234567890",
        "SOME_VENDOR_SECRET": "sec-1234567890",
        "SOME_VENDOR_URL": "https://example.invalid",
    }
    shown = obfuscate_config_for_display(config)

    for key in ("SOME_VENDOR_API_KEY", "SOME_VENDOR_PASSWORD", "SOME_VENDOR_TOKEN", "SOME_VENDOR_SECRET"):
        assert shown[key] != config[key], key
    assert shown["SOME_VENDOR_URL"] == config["SOME_VENDOR_URL"]


def test_input_is_not_mutated():
    config = {"OPENAI_API_KEY": "sk-openai-1234567890"}
    obfuscate_config_for_display(config)
    assert config["OPENAI_API_KEY"] == "sk-openai-1234567890"


def _run_config(monkeypatch, capsys, show_secrets, json_out=False):
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-1234567890")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-gemini-1234567890")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    cli_standalone.cmd_config(argparse.Namespace(show_secrets=show_secrets, json_out=json_out))
    return capsys.readouterr().out


def test_cli_config_masks_provider_keys_by_default(monkeypatch, capsys):
    out = _run_config(monkeypatch, capsys, show_secrets=False)
    assert "GEMINI_API_KEY=AIza" in out
    assert "sk-openai-1234567890" not in out
    assert "AIza-gemini-1234567890" not in out


def test_cli_config_show_secrets_reveals_keys(monkeypatch, capsys):
    out = _run_config(monkeypatch, capsys, show_secrets=True)
    assert "OPENAI_API_KEY=sk-openai-1234567890" in out
    assert "GEMINI_API_KEY=AIza-gemini-1234567890" in out


def test_explicit_mask_list_still_applies_without_a_secret_suffix():
    config = {
        "ZOTERO_LIBRARY_ID": "1234567",
        "ZOTERO_WEBDAV_URL": "https://dav.example.invalid/zotero",
        "ZOTERO_WEBDAV_USERNAME": "someone",
        "API_KEY": "bare-key-1234567890",
        "LIBRARY_ID": "7654321",
        "WEBDAV_URL": "https://dav.example.invalid/other",
        "WEBDAV_USERNAME": "someone-else",
    }
    shown = obfuscate_config_for_display(config)
    for key, value in config.items():
        assert shown[key] != value, key
        assert shown[key].startswith(value[:4]), key


def test_cli_config_json_masks_provider_keys(monkeypatch, capsys):
    out = _run_config(monkeypatch, capsys, show_secrets=False, json_out=True)
    envelope = json.loads(out)
    settings = envelope["data"]["settings"]

    for key, value in (
        ("OPENAI_API_KEY", "sk-openai-1234567890"),
        ("GEMINI_API_KEY", "AIza-gemini-1234567890"),
    ):
        shown = settings[key]
        assert shown != value, key
        assert shown.startswith(value[:4]), key
        assert set(shown[4:]) == {"*"}, key
