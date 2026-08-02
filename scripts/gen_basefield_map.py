#!/usr/bin/env python3
"""Regenerate the vendored Zotero base-field map shipped with the package.

The map is a trimmed slice of Zotero's global schema
(https://api.zotero.org/schema) carrying, per item type, the mapping from each
type-specific field key to its Zotero *base* field. It lets ``zotero_mcp``
resolve generic parameters (``title``, ``date``, ``publisher``, ...) to the
actual field a given item type uses (a statute's ``title`` is ``nameOfAct``)
and validate a field against the type's declared field set rather than against
the presence of that key on a fetched item.

Run this whenever Zotero bumps the schema version:

    python scripts/gen_basefield_map.py            # fetch live schema
    python scripts/gen_basefield_map.py PATH.json  # use a local schema copy

Output is written to ``src/zotero_mcp/data/zotero_basefields.json``. The weekly
CI job diffs that file and opens a PR when the version changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_URL = "https://api.zotero.org/schema"
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "zotero_mcp" / "data" / "zotero_basefields.json"
)


def _load_schema(source: str | None) -> dict:
    if source:
        return json.loads(Path(source).read_text(encoding="utf-8"))
    import requests

    resp = requests.get(SCHEMA_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_map(schema: dict) -> dict:
    """Reduce the full schema to ``{version, itemTypes: {type: {field: base}}}``.

    ``base`` defaults to the field's own name when the field is not a rename,
    so ``itemTypes[type]`` doubles as the type's full valid-field set (its keys)
    and its base->actual inverse (invert the mapping).
    """
    item_types: dict[str, dict[str, str]] = {}
    for it in schema["itemTypes"]:
        fields = {
            f["field"]: f.get("baseField", f["field"])
            for f in it.get("fields", [])
        }
        item_types[it["itemType"]] = fields
    return {
        "version": schema["version"],
        "itemTypes": dict(sorted(item_types.items())),
    }


def main(argv: list[str]) -> int:
    schema = _load_schema(argv[1] if len(argv) > 1 else None)
    table = build_map(schema)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {OUT_PATH.relative_to(Path.cwd())} "
        f"(schema version {table['version']}, {len(table['itemTypes'])} types)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
