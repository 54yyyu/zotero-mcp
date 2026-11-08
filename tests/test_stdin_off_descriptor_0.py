"""On Windows over stdio, stdin is read through a duplicate of descriptor 0 (#485).

The thread blocked in the stdio server's read of stdin holds the C runtime's
lock on descriptor 0, and the OpenBLAS DLL that numpy and scipy bundle touches
descriptor 0 while it initialises, so loading it (importing numpy, scipy.linalg
or scikit-learn, as sentence-transformers does at the first semantic search)
waited for the next message on stdin. ``serve`` now reads the MCP stream
through a duplicate descriptor and points descriptor 0 at NUL.
"""

import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


_STDIN_PROBE = """
import os, stat, sys, threading
sys.path.insert(0, {src!r})
from zotero_mcp.cli import _read_stdin_through_a_duplicate_descriptor

moved = _read_stdin_through_a_duplicate_descriptor()
line = []
reader = threading.Thread(target=lambda: line.append(sys.stdin.buffer.readline()), daemon=True)
reader.start()  # blocks on the pipe, as the MCP stdio reader does
imported = threading.Event()

def load():
    import numpy  # noqa: F401  (its OpenBLAS DLL is what hung in #485)
    imported.set()

threading.Thread(target=load, daemon=True).start()
ok = imported.wait({wait})
print("moved", moved, "imported", ok, "fd0_is_a_device", stat.S_ISCHR(os.fstat(0).st_mode), flush=True)
reader.join({wait})
print("line", line[0].decode().strip() if line else None, flush=True)
"""


def test_stdin_moves_off_descriptor_0_and_still_delivers_messages():
    """The MCP stream keeps working, and a DLL load no longer waits for stdin.

    On Windows, without the move, the numpy import below waits until the next
    line arrives on stdin (#485), and so does anything else that touches
    descriptor 0; on other systems it never waited. Timed to fail well inside
    the suite's 30 s per-test limit rather than hang.
    """
    pytest.importorskip("numpy")
    proc = subprocess.Popen(
        [sys.executable, "-c", _STDIN_PROBE.format(src=SRC, wait=15)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(out) for out in proc.stdout], daemon=True).start()
    try:
        first = lines.get(timeout=20).split()
        proc.stdin.write('{"jsonrpc": "2.0", "method": "ping"}\n')
        proc.stdin.flush()
        second = lines.get(timeout=5).strip()
        proc.wait(timeout=5)
    except queue.Empty:
        pytest.fail("the probe stopped answering: a DLL load waited for stdin (#485)")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    assert proc.returncode == 0, proc.stderr.read()
    assert first == ["moved", "True", "imported", "True", "fd0_is_a_device", "True"]
    assert second == 'line {"jsonrpc": "2.0", "method": "ping"}'
