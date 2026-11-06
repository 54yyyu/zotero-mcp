"""Records that failed to embed are written again, without stalling updates (#769)."""

import time

from zotero_mcp import batch_common

# --- add_failed_ids ---------------------------------------------------------------


def test_add_failed_ids_keeps_order_and_drops_repeats():
    manifest = {"failed_ids": ["B", "A"]}

    batch_common.add_failed_ids(manifest, ["A", "C", "C", "D"])

    assert manifest["failed_ids"] == ["B", "A", "C", "D"]


def test_add_failed_ids_is_linear():
    """A large run gives up whole files of ids; a quadratic check takes minutes."""
    manifest = {"failed_ids": [f"old{i}" for i in range(30000)]}
    started = time.monotonic()

    batch_common.add_failed_ids(manifest, [f"new{i}" for i in range(30000)])

    assert time.monotonic() - started < 2
    assert len(manifest["failed_ids"]) == 60000
