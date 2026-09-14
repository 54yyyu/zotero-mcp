#!/usr/bin/env python3
"""Profile where `recent_items` spends its time on the SQLite backend.

`measure_read_backend.py` counts how much each read costs. This shows where the
time in one of them goes, which is the question to answer when `recent_items`
is slow on a large library: the SQL, the Python that builds the page, or a WAL
snapshot refresh.

For `SqliteBackend.recent_items(limit)` it reports:

1. first-call and warm timings, and each SQL statement's share of them;
2. EXPLAIN QUERY PLAN for every statement;
3. for the slowest statement, its time with and without ORDER BY ... LIMIT,
   which separates the sort from the scan that feeds it;
4. the same call through `get_library_backend()`, the path the tools use,
   including its per-call staleness check;
5. what a snapshot refresh costs here: copying zotero.sqlite, then the first
   call against the copy (skipped when the temp directory lacks room);
6. cProfile of the first call and of the warm calls.

    python scripts/profile_recent_items.py                    # personal library
    python scripts/profile_recent_items.py --group-id 123456  # one group library
    python scripts/profile_recent_items.py --all-libraries    # global scope
    python scripts/profile_recent_items.py --dump-dir out/    # also write .prof files

Needs a readable zotero.sqlite; set ZOTERO_DB_PATH if Zotero's data directory
is somewhere custom. Zotero does not need to be running. Reads only: the
snapshot test copies the database into a temporary directory and deletes the
copy afterwards.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import platform
import pstats
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

#: Page size, the same as measure_read_backend.py's batch.
DEFAULT_LIMIT = 25

#: Calls per timing. Enough for a stable median, few enough to finish in
#: seconds on a large library.
DEFAULT_RUNS = 20


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:,.1f} ms"


def _one_line(sql: str, width: int = 90) -> str:
    text = " ".join(sql.split())
    return text if len(text) <= width else text[: width - 1] + "…"


class _TimedCursor:
    def __init__(self, cursor, record):
        self._cursor = cursor
        self._record = record

    def _timed(self, fetch, *args):
        started = time.perf_counter()
        result = fetch(*args)
        self._record["fetch"] += time.perf_counter() - started
        return result

    def fetchall(self):
        rows = self._timed(self._cursor.fetchall)
        self._record["rows"] += len(rows)
        return rows

    def fetchone(self):
        row = self._timed(self._cursor.fetchone)
        self._record["rows"] += row is not None
        return row

    def fetchmany(self, *args):
        rows = self._timed(self._cursor.fetchmany, *args)
        self._record["rows"] += len(rows)
        return rows

    def __iter__(self):
        while (row := self.fetchone()) is not None:
            yield row

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _TimedConnection:
    """Proxy that records each statement's execute time, fetch time and rows.

    Like measure_read_backend.py's counter, it wraps the reader's connection
    rather than patching sqlite3, so the reader itself runs unchanged.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "log", [])

    def execute(self, sql, params=()):
        record = {"sql": sql, "params": params, "exec": 0.0, "fetch": 0.0, "rows": 0}
        started = time.perf_counter()
        cursor = self._inner.execute(sql, params)
        record["exec"] = time.perf_counter() - started
        self.log.append(record)
        return _TimedCursor(cursor, record)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


def _plan(conn, sql, params) -> list[str]:
    return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()]


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%h %s"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown (git not available)"
    return out.stdout.strip() or "unknown (not a git checkout)"


def _wal_bytes(db_path: str) -> int:
    wal = os.path.realpath(db_path) + "-wal"
    return os.path.getsize(wal) if os.path.exists(wal) else 0


def _print_environment(group_id: int | None, collection: str | None) -> None:
    import zotero_mcp
    from zotero_mcp.local_db import get_local_zotero_reader

    reader = get_local_zotero_reader()
    if reader is None:
        raise SystemExit(
            "No readable zotero.sqlite. Set ZOTERO_DB_PATH if your Zotero data "
            "directory is in a custom location."
        )
    try:
        conn = reader._get_connection()
        lib_ids = reader._resolve_scope_library_ids(group_id)
        if not lib_ids:
            raise SystemExit(f"No library for group_id={group_id} in this database.")
        placeholders = ",".join("?" * len(lib_ids))
        in_scope = conn.execute(
            f"SELECT COUNT(*) FROM items WHERE libraryID IN ({placeholders})", lib_ids
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        target = getattr(reader, "_connection_target", ("unknown",))[0]
        db_path = os.path.realpath(reader.db_path)
    finally:
        reader.close()

    scope = "all libraries" if group_id is None else (
        "personal library" if group_id == 0 else f"group {group_id}"
    )
    narrowed = f", narrowed to collection {collection}" if collection else ""
    print("== Environment")
    print(f"code:      {_git_revision()} (zotero_mcp {getattr(zotero_mcp, '__version__', 'unknown')})")
    print(f"runtime:   Python {platform.python_version()}, SQLite {sqlite3.sqlite_version}, {platform.platform()}")
    print(f"database:  {os.path.getsize(db_path) / 1e6:,.0f} MB, WAL {_wal_bytes(db_path):,} bytes, "
          f"read {'from a snapshot copy' if target == 'snapshot' else 'in place'}")
    print(f"scope:     {scope}: {in_scope:,} of {total:,} items rows ({in_scope / total if total else 0:.0%}){narrowed}")


def _time_calls(backend, kwargs, runs, timed=None):
    totals, logs = [], []
    for _ in range(runs):
        if timed is not None:
            timed.log.clear()
        started = time.perf_counter()
        backend.recent_items(**kwargs)
        totals.append(time.perf_counter() - started)
        if timed is not None:
            logs.append(list(timed.log))
    return totals, logs


def _by_statement(log) -> dict[str, list]:
    """{normalized SQL: [seconds, rows, first record]} for one call."""
    out: dict[str, list] = {}
    for record in log:
        key = " ".join(record["sql"].split())
        entry = out.setdefault(key, [0.0, 0, record])
        entry[0] += record["exec"] + record["fetch"]
        entry[1] += record["rows"]
    return out


def _print_statements(cold_log, warm_logs, warm_totals, runs):
    """Per-statement timings, matched by SQL text rather than position.

    Some statements run only on a connection's first call (library labels are
    cached after it), so a statement's position is not a stable identity.
    Returns [(record, warm median)] in first-seen order.
    """
    cold = _by_statement(cold_log)
    warm = [_by_statement(log) for log in warm_logs]
    keys = list(cold)
    for call in warm:
        keys.extend(k for k in call if k not in keys)

    median_total = statistics.median(warm_totals)
    print(f"\n{'#':>2} {'rows':>6} {'first call':>11} {'warm median':>12} {'warm calls':>10} {'share':>6}  statement")
    statements = []
    for n, key in enumerate(keys, 1):
        samples = [call[key][0] for call in warm if key in call]
        warm_median = statistics.median(samples) if samples else 0.0
        record = cold[key][2] if key in cold else next(call[key][2] for call in warm if key in call)
        first = _ms(cold[key][0]) if key in cold else "—"
        rows = cold[key][1] if key in cold else "—"
        share = warm_median / median_total if median_total else 0.0
        print(f"{n:>2} {rows:>6} {first:>11} {_ms(warm_median):>12} {f'{len(samples)}/{runs}':>10} "
              f"{share:>6.0%}  {_one_line(record['sql'])}")
        statements.append((record, warm_median))

    sql_median = statistics.median(sum(entry[0] for entry in call.values()) for call in warm)
    print(f"   all SQL, warm median {_ms(sql_median)}; everything else {_ms(median_total - sql_median)}")
    return statements


def _sort_versus_scan(conn, record, runs) -> None:
    """Time a statement with and without its ORDER BY ... LIMIT.

    Without the sort, the statement is wrapped in COUNT(*) so SQLite still has
    to visit every row that passes the filters. The two only measure the same
    scan if their plans agree once the sort's temp B-tree is set aside, so that
    is checked and reported.
    """
    sql, params = record["sql"], record["params"]
    match = re.search(r"\bORDER BY\b[\s\S]*?\bLIMIT \?", sql)
    if not isinstance(params, (list, tuple)) or match is None or "?" in sql[match.end():]:
        print("the slowest statement has no trailing ORDER BY ... LIMIT ?, so there is no sort to separate")
        return
    params = list(params)
    count_sql = f"SELECT COUNT(*) FROM ({sql[:match.start()]}{sql[match.end():]})"
    count_params = params[:-1]

    def without_sort_step(plan):
        return [step for step in plan if "TEMP B-TREE" not in step]

    same_scan = without_sort_step(_plan(conn, sql, params)) == without_sort_step(_plan(conn, count_sql, count_params))

    sorted_times, scan_times = [], []
    for _ in range(runs):  # interleaved, so drift affects both equally
        started = time.perf_counter()
        conn.execute(sql, params).fetchall()
        sorted_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        conn.execute(count_sql, count_params).fetchall()
        scan_times.append(time.perf_counter() - started)

    candidates = conn.execute(count_sql, count_params).fetchone()[0]
    with_sort, scan_only = statistics.median(sorted_times), statistics.median(scan_times)
    print(f"rows passing the filters, i.e. sorted: {candidates:,}")
    print(f"as written:                {_ms(with_sort)} median of {runs}")
    print(f"same filters, no sort:     {_ms(scan_only)} median of {runs}")
    print(f"sort share:                about {_ms(max(0.0, with_sort - scan_only))}")
    if not same_scan:
        print("note: the unsorted query took a different plan, so this split is only approximate")


def _server_path(kwargs, runs, group_id) -> None:
    from zotero_mcp.client import get_active_group_id
    from zotero_mcp.library import get_library_backend, reset_sqlite_reader

    if group_id is None:
        print("skipped for --all-libraries: get_library_backend() reads one library, the active one")
        return

    # get_library_backend() reads the active library, which these variables
    # select. Point them at the library profiled above, so both sections time
    # the same scope.
    saved = {name: os.environ.get(name) for name in ("ZOTERO_LIBRARY_ID", "ZOTERO_LIBRARY_TYPE")}
    if group_id:
        os.environ["ZOTERO_LIBRARY_ID"] = str(group_id)
        os.environ["ZOTERO_LIBRARY_TYPE"] = "group"
    else:
        os.environ["ZOTERO_LIBRARY_TYPE"] = "user"
    times = []
    try:
        active = get_active_group_id()
        backend_name = type(get_library_backend()).__name__
        for _ in range(runs + 1):
            started = time.perf_counter()
            get_library_backend().recent_items(**kwargs)
            times.append(time.perf_counter() - started)
    finally:
        reset_sqlite_reader()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print(f"backend: {backend_name}, scope group_id={active}")
    if active != group_id:
        print("note: the active library could not be set to the profiled one, so compare with care")
    print(f"first call, which opens the thread's reader: {_ms(times[0])}")
    print(f"next {runs} calls, each checking refresh_if_stale: median {_ms(statistics.median(times[1:]))}, "
          f"max {_ms(max(times[1:]))}")


def _snapshot_cost(db_path, group_id, kwargs, skip) -> None:
    from zotero_mcp.library import SqliteBackend
    from zotero_mcp.local_db import DB_SNAPSHOT_ENV_VAR, LocalZoteroReader

    if skip:
        print("skipped (--skip-snapshot)")
        return
    source = os.path.realpath(db_path)
    wal = source + "-wal"
    needed = os.path.getsize(source) + _wal_bytes(source)
    temp_dir = tempfile.gettempdir()
    free = shutil.disk_usage(temp_dir).free
    if free < 2 * needed:
        print(f"skipped: copying needs {needed / 1e6:,.0f} MB and {temp_dir} has {free / 1e6:,.0f} MB free")
        return

    snap_dir = tempfile.mkdtemp(prefix="zotero_mcp_profile_")
    previous = os.environ.get(DB_SNAPSHOT_ENV_VAR)
    try:
        snap = os.path.join(snap_dir, "zotero.sqlite")
        started = time.perf_counter()
        shutil.copyfile(source, snap)
        if _wal_bytes(source):
            shutil.copyfile(wal, snap + "-wal")
        copied = time.perf_counter() - started

        # Read the copy in place; otherwise the reader would snapshot the snapshot.
        os.environ[DB_SNAPSHOT_ENV_VAR] = "0"
        reader = LocalZoteroReader(db_path=snap)
        try:
            started = time.perf_counter()
            SqliteBackend(reader, group_id).recent_items(**kwargs)
            first_call = time.perf_counter() - started
        finally:
            reader.close()
        print(f"copy {needed / 1e6:,.0f} MB of database and WAL: {_ms(copied)}")
        print(f"first call against the fresh copy:   {_ms(first_call)}")
    except OSError as exc:
        print(f"skipped: could not copy the database ({exc})")
    finally:
        if previous is None:
            os.environ.pop(DB_SNAPSHOT_ENV_VAR, None)
        else:
            os.environ[DB_SNAPSHOT_ENV_VAR] = previous
        shutil.rmtree(snap_dir, ignore_errors=True)


def _print_profile(profile, title, dump_dir, filename, top=18) -> None:
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        profile.dump_stats(dump_dir / filename)
    for key in ("cumulative", "tottime"):
        buffer = io.StringIO()
        pstats.Stats(profile, stream=buffer).strip_dirs().sort_stats(key).print_stats(top)
        text = buffer.getvalue()
        start = text.find("   ncalls")
        print(f"\n--- {title}, sorted by {key} (top {top}) ---")
        print((text[start:] if start >= 0 else text).rstrip())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"page size passed to recent_items (default {DEFAULT_LIMIT})")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"calls per timing (default {DEFAULT_RUNS})")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--group-id", type=int, default=0,
                       help="Zotero groupID to profile (default 0, the personal library)")
    scope.add_argument("--all-libraries", action="store_true",
                       help="profile the global scope: every user and group library")
    parser.add_argument("--collection", metavar="KEY",
                        help="restrict to one collection, as recent_items(collection_key=...) does")
    parser.add_argument("--skip-snapshot", action="store_true",
                        help="don't time a copy of zotero.sqlite")
    parser.add_argument("--dump-dir", type=Path,
                        help="also write the cProfile data here as .prof files")
    args = parser.parse_args()

    os.environ.setdefault("ZOTERO_LOCAL", "true")
    os.environ["ZOTERO_BACKEND"] = "sqlite"

    from zotero_mcp.library import SqliteBackend
    from zotero_mcp.local_db import get_local_zotero_reader

    runs = max(1, args.runs)
    group_id = None if args.all_libraries else args.group_id
    kwargs = {"limit": max(1, args.limit), "collection_key": args.collection}

    _print_environment(group_id, args.collection)

    # A fresh connection, opened before timing as measure_read_backend.py does.
    reader = get_local_zotero_reader()
    db_path = reader.db_path
    reader._get_connection()
    cold_profile = cProfile.Profile()
    started = time.perf_counter()
    cold_profile.runcall(SqliteBackend(reader, group_id).recent_items, **kwargs)
    cold_profiled = time.perf_counter() - started
    reader.close()

    reader = get_local_zotero_reader()
    conn = reader._get_connection()
    timed = _TimedConnection(conn)
    reader._connection = timed
    backend = SqliteBackend(reader, group_id)
    try:
        started = time.perf_counter()
        backend.recent_items(**kwargs)
        cold_total = time.perf_counter() - started
        cold_log = list(timed.log)
        warm_totals, warm_logs = _time_calls(backend, kwargs, runs, timed)

        print(f"\n== recent_items(limit={kwargs['limit']}) on SqliteBackend, connection already open")
        print(f"first call on a new connection: {_ms(cold_total)} (another, under cProfile: {_ms(cold_profiled)})")
        print(f"warm, {runs} calls: median {_ms(statistics.median(warm_totals))}, "
              f"min {_ms(min(warm_totals))}, max {_ms(max(warm_totals))}")
        statements = _print_statements(cold_log, warm_logs, warm_totals, runs)

        reader._connection = conn
        print("\n== EXPLAIN QUERY PLAN")
        for n, (record, _) in enumerate(statements, 1):
            print(f"\n#{n} {_one_line(record['sql'], 80)}")
            for step in _plan(conn, record["sql"], record["params"]):
                print(f"     {step}")

        slowest = max(statements, key=lambda item: item[1])[0]
        print(f"\n== Sort versus scan, slowest statement: {_one_line(slowest['sql'], 70)}")
        _sort_versus_scan(conn, slowest, runs)

        warm_profile = cProfile.Profile()
        warm_profile.enable()
        _time_calls(backend, kwargs, runs)
        warm_profile.disable()
    finally:
        reader._connection = conn
        reader.close()

    print("\n== The tools' path: get_library_backend().recent_items()")
    _server_path(kwargs, runs, group_id)

    print("\n== Cost of a snapshot refresh (made when Zotero's WAL holds changes)")
    _snapshot_cost(db_path, group_id, kwargs, args.skip_snapshot)

    print(f"\nWAL at the end of the run: {_wal_bytes(db_path):,} bytes")
    _print_profile(cold_profile, "cProfile, first call on a new connection", args.dump_dir, "recent_items_first_call.prof")
    _print_profile(warm_profile, f"cProfile, {runs} warm calls", args.dump_dir, "recent_items_warm.prof")


if __name__ == "__main__":
    main()
