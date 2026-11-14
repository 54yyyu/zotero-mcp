"""Regenerate pre017.zip and v0171.zip: real indexes written by the real old releases.

    uv run --no-project python tests/fixtures/legacy_indexes/make_legacy_indexes.py

Not run in CI; the zips are committed. Each release runs in its own throwaway ``uv`` environment
(pinned zotero-mcp-server and chromadb), with HOME and every temporary file in a temp directory.
Only the PyPI install needs the network; the builds themselves are offline (see ``_build.py``).
See README.md for the versions used and why.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

PYTHON = "3.11"
CHROMADB = "1.5.9"  # the latest 1.x when 0.16.0 and 0.17.1 shipped (2026-11), and uv.lock's
BUILDS = {
    # zip name: zotero-mcp-server release
    "pre017.zip": "0.16.0",  # latest 0.16.x; one collection, passages only (chunking on)
    "v0171.zip": "0.17.1",  # articles in zotero_library, passages in zotero_library_passages
}


def _clean_env(home: Path) -> dict[str, str]:
    """No inherited virtualenv, config or credentials; HOME inside the temp directory."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:" + os.path.dirname(shutil.which("uv") or "/usr/local/bin/uv"),
        "HOME": str(home),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(Path(os.path.expanduser("~")) / ".cache" / "uv"),
        "LANG": "C.UTF-8",
    }
    return env


def build(zip_name: str, version: str, work: Path) -> Path:
    work.mkdir(parents=True)
    (work / "home").mkdir()
    cmd = [
        "uv", "run", "--no-project", "--isolated", "--python", PYTHON,
        "--with", f"zotero-mcp-server=={version}", "--with", f"chromadb=={CHROMADB}",
        "python", "-I", str(HERE / "_build.py"), str(work),
    ]  # fmt: skip
    # -I drops the script directory from sys.path, so _build.py adds HERE itself.
    subprocess.run(cmd, check=True, env=_clean_env(work / "home"), cwd=work)
    config_dir = work / "home" / ".config" / "zotero-mcp"
    out = HERE / zip_name
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(config_dir.rglob("*")):
            if path.is_file() and path.name != "update.lock":  # a runtime lock, not index state
                info = zipfile.ZipInfo(str(path.relative_to(config_dir)), date_time=(2026, 11, 14, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, path.read_bytes())
    return out


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="legacy-indexes-") as tmp:
        for zip_name, version in BUILDS.items():
            out = build(zip_name, version, Path(tmp) / version)
            print(f"{out.name}: zotero-mcp-server {version}, {out.stat().st_size / 1000:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
