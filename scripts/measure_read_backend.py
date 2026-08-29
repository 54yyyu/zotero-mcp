#!/usr/bin/env python3
"""Measure what each read backend costs to answer the same question.

`zotero_mcp.library` serves every read operation through one of two
implementations: `SqliteBackend`, which queries `zotero.sqlite` directly, and
`ApiBackend`, which makes the pyzotero calls the tools used to make. They
return the same records; what differs is the work.

This counts that work rather than estimating it. SQL statements are counted by
proxying the reader's sqlite3 connection; HTTP requests by wrapping
`httpx.Client.send`. Both counts are observed at the point of execution, so a
retry, a redirect or a pagination loop shows up as what it is.

The number that matters here is not per-call latency -- it is how each column
*scales*. The SQLite column is flat: an operation costs the same handful of
queries whether it is asked for one key or a hundred. The API column tracks
either the batch size (one request per parent for children) or the size of the
library (one request per 100 rows for tags). That shape is what the port
exists to remove, and it is why the ratios below grow with the library rather
than staying constant.

What this deliberately does NOT measure: correctness or agreement between the
backends -- that is `tests/live/test_read_backend_parity.py`, which compares
their actual results. A fast backend that returns the wrong rows would look
excellent here, so the two are kept separate on purpose.

Both routes are measured when both are available. With Zotero desktop closed
the API column is reported as unavailable rather than skipped silently: those
reads working at all is the point of the exercise.

    python scripts/measure_read_backend.py              # table
    python scripts/measure_read_backend.py --json       # machine-readable
    python scripts/measure_read_backend.py --keys 100   # bigger batch

Needs ZOTERO_LOCAL=true and a readable zotero.sqlite. Reads only; it never
writes to the database or to Zotero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

#: Batch size for the plural operations, unless --keys says otherwise. Large
#: enough that an N+1 shape is unmistakable, small enough to stay polite to a
#: live API.
DEFAULT_KEYS = 25


class _CountingConnection:
    """Proxy that counts `execute` calls on a sqlite3 connection.

    sqlite3.Connection.execute is read-only, so it cannot be patched in
    place; wrapping is the only way to observe the statement count without
    changing the reader.
    """

    def __init__(self, inner, counter):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_counter", counter)

    def execute(self, *args, **kwargs):
        self._counter["sql"] += 1
        return self._inner.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_counters(reader, counter):
    """Count SQL statements and HTTP requests for the rest of the process."""
    import httpx

    reader._connection = _CountingConnection(reader._get_connection(), counter)

    original_send = httpx.Client.send

    def counting_send(self, request, *args, **kwargs):
        counter["http"] += 1
        return original_send(self, request, *args, **kwargs)

    httpx.Client.send = counting_send


def _operations(keys):
    """(label, callable) pairs, each taking a backend.

    Chosen to cover the three shapes that behave differently: a point lookup,
    a batch keyed on N, and a listing keyed on library size.
    """
    return [
        ("get_item(1 key)", lambda b: b.get_item(keys[0])),
        (f"get_items({len(keys)} keys)", lambda b: b.get_items(keys)),
        (f"get_children({len(keys)} keys)", lambda b: b.get_children(keys)),
        ("list_collections()", lambda b: b.list_collections()),
        ("list_tags()", lambda b: b.list_tags()),
        (f"recent_items({len(keys)})", lambda b: b.recent_items(limit=len(keys))),
    ]


def _measure(backend, operation, counter, kind):
    """Run one operation, returning its cost or the reason it could not run."""
    counter["sql"] = counter["http"] = 0
    started = time.perf_counter()
    try:
        operation(backend)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:110]}
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"calls": counter[kind], "ms": round(elapsed_ms, 1)}


def collect(key_count):
    """Measure every operation on whichever backends are usable here."""
    os.environ.setdefault("ZOTERO_LOCAL", "true")
    os.environ["ZOTERO_BACKEND"] = "sqlite"

    from zotero_mcp.client import get_local_zotero_client
    from zotero_mcp.library import ApiBackend, SqliteBackend
    from zotero_mcp.local_db import get_local_zotero_reader

    reader = get_local_zotero_reader()
    if reader is None:
        raise SystemExit(
            "No readable zotero.sqlite. Set ZOTERO_LOCAL=true, and ZOTERO_DB_PATH "
            "if your Zotero data directory is in a custom location."
        )

    counter = {"sql": 0, "http": 0}
    _install_counters(reader, counter)

    sqlite_backend = SqliteBackend(reader, 0)
    keys = [item["key"] for item in sqlite_backend.recent_items(limit=key_count)]
    if not keys:
        raise SystemExit("The personal library has no items to measure against.")

    zot = get_local_zotero_client()
    api_backend = ApiBackend(zot) if zot is not None else None

    rows = []
    for label, operation in _operations(keys):
        row = {
            "operation": label,
            "sqlite": _measure(sqlite_backend, operation, counter, "sql"),
        }
        row["api"] = (
            _measure(api_backend, operation, counter, "http")
            if api_backend is not None
            else {"unavailable": "Zotero API not reachable"}
        )
        rows.append(row)

    result = {
        "library_items": reader.get_item_count(),
        "keys_measured": len(keys),
        "api_available": api_backend is not None,
        "operations": rows,
    }
    reader.close()
    return result


def _cell(measurement, unit):
    if "error" in measurement:
        return "error", "—"
    if "unavailable" in measurement:
        return "—", "—"
    return f"{measurement['calls']} {unit}", f"{measurement['ms']:,.1f} ms"


def render(result):
    lines = []
    add = lines.append

    add(f"Library: {result['library_items']:,} items · "
        f"batch size: {result['keys_measured']} keys")
    if not result["api_available"]:
        add("")
        add("Zotero API not reachable — SQLite column only. Every operation below")
        add("is answered with no network access at all, which is the point: these")
        add("reads used to be impossible with Zotero desktop closed.")
    add("")

    header = f"{'operation':<26}{'sqlite':>10}{'':>13}{'api':>12}{'':>15}"
    add(header)
    add(f"{'':<26}{'queries':>10}{'time':>13}{'requests':>12}{'time':>15}")
    add("-" * len(header))

    for row in result["operations"]:
        s_calls, s_ms = _cell(row["sqlite"], "")
        a_calls, a_ms = _cell(row["api"], "")
        add(f"{row['operation']:<26}{s_calls:>10}{s_ms:>13}{a_calls:>12}{a_ms:>15}")

    if result["api_available"]:
        add("")
        add("The sqlite column is flat in batch size; the api column is not.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of a table")
    parser.add_argument("--keys", type=int, default=DEFAULT_KEYS,
                        help=f"how many item keys to batch (default {DEFAULT_KEYS})")
    args = parser.parse_args()

    result = collect(max(1, args.keys))
    print(json.dumps(result, indent=2) if args.json else render(result))


if __name__ == "__main__":
    main()
