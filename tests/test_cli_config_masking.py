"""`zotero-cli config` prints resolved settings, and the packaged skill tells
agents to run it first, so every key-shaped value must be masked by default.
The mask list used to name only Zotero/WebDAV keys; OPENAI_API_KEY and
GOOGLE_API_KEY were printed in full.
"""

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
