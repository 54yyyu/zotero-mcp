"""Upgrade-path details of the item-keyed index (#826, final review I1).

I1: new-shape writes send ``None`` values, which chromadb 1.0.0 and 1.0.1 reject.
"""

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

ROOT = Path(__file__).resolve().parent.parent


def test_the_chromadb_floor_accepts_none_metadata_values():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    spec = re.search(r'"chromadb>=(\d+)\.(\d+)\.(\d+)"', pyproject)

    assert spec, "chromadb has a lower bound"
    assert tuple(int(part) for part in spec.groups()) >= (1, 0, 2)
