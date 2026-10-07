"""A Windows console must not crash the CLI on a character it cannot encode (#26).

``setup-info`` prints emoji and ``zotero-cli`` prints item titles in any
script; under cp1252 or cp936 ``print`` raised UnicodeEncodeError and aborted
the command, so ``setup`` died before it wrote its config.
"""

import io
import sys
from unittest.mock import patch

import pytest

from zotero_mcp import cli, cli_standalone


def _console(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    return raw


@pytest.mark.parametrize(
    "entry, argv",
    [
        (cli.main, ["zotero-mcp", "help"]),
        (cli_standalone.main, ["zotero-cli", "--json-schema"]),
    ],
)
def test_entry_points_survive_a_console_that_cannot_encode(monkeypatch, entry, argv):
    raw = _console(monkeypatch)
    with patch("sys.argv", argv), pytest.raises(SystemExit):
        entry()
    print("\U0001f527 Installation Details: → 论文")  # must not raise
    assert raw.getvalue().endswith(b"Installation Details: ? ??\n")
