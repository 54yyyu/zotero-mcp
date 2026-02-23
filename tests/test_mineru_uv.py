#!/usr/bin/env python3
"""Local integration runner for MinerU vectorization flow.

This script replaces `scripts/test_mineru_uv.sh` with a Python equivalent.
It is intended for manual local verification, not CI execution.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run MinerU local integration checks via zotero_mcp CLI."
    )
    parser.add_argument(
        "--env-file",
        default=str(project_root / ".env"),
        help="Path to .env file (default: <project>/.env)",
    )
    parser.add_argument(
        "--limit",
        default=20,
        type=int,
        help="Limit items for update-db test (default: 20)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Add --force-rebuild to update-db",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="Only run doctor + db-status, skip update-db",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="Pass --config-path to CLI commands",
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="Pass --db-path to update-db",
    )
    parser.add_argument(
        "--no-auto-config",
        action="store_true",
        help="Disable auto-generated project test config",
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help="Force using 'uv run python -m zotero_mcp.cli'",
    )
    parser.add_argument(
        "--python",
        dest="python_exe",
        default="",
        help="Python executable for cli module (default: .venv/bin/python)",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f".env file not found at: {path}")

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def resolve_runner(project_root: Path, python_exe: str, use_uv: bool) -> list[str]:
    resolved_python = python_exe or str(project_root / ".venv" / "bin" / "python")
    uv_bin = shutil.which("uv")

    if use_uv:
        if not uv_bin:
            raise RuntimeError("--use-uv specified but uv is not installed or not in PATH.")
        return ["uv", "run", resolved_python, "-m", "zotero_mcp.cli"]

    if Path(resolved_python).exists():
        return [resolved_python, "-m", "zotero_mcp.cli"]
    if uv_bin:
        return ["uv", "run", resolved_python, "-m", "zotero_mcp.cli"]
    raise RuntimeError(f"python executable not found: {resolved_python}")


def token_count() -> int:
    tokens = os.getenv("MINERU_TOKENS", "").strip()
    if tokens:
        return len([t for t in [x.strip() for x in tokens.split(",")] if t])
    return 1 if os.getenv("MINERU_TOKEN", "").strip() else 0


def ensure_mineru_token() -> int:
    if (
        os.getenv("MINERU_API_TOKEN", "").strip()
        and not os.getenv("MINERU_TOKEN", "").strip()
        and not os.getenv("MINERU_TOKENS", "").strip()
    ):
        os.environ["MINERU_TOKEN"] = os.environ["MINERU_API_TOKEN"].strip()

    count = token_count()
    if count <= 0:
        raise RuntimeError(
            "No MinerU token found. Set MINERU_TOKEN, MINERU_TOKENS, or MINERU_API_TOKEN."
        )
    return count


def build_auto_config(project_root: Path) -> Path:
    tmp_dir = project_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / "mineru-test-config.json"
    locator_db_path = tmp_dir / "locator.db"
    md_store_dir = tmp_dir / "md_store"
    chroma_db_dir = tmp_dir / "chroma_db"

    config = {
        "semantic_search": {
            "embedding_model": "default",
            "collection_name": "zotero_library_chunks_v2",
            "persist_directory": str(chroma_db_dir),
            "extraction_mode": "mineru",
            "mineru": {"enabled": True},
            "locator_db": {"path": str(locator_db_path)},
            "md_store": {"base_dir": str(md_store_dir)},
        }
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def run_cli(project_root: Path, runner: list[str], args: list[str]) -> None:
    cmd = runner + args
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    printable = " ".join(shlex.quote(x) for x in (["env", f"PYTHONPATH={env['PYTHONPATH']}"] + cmd))
    print()
    print(f"Running: {printable}")
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    ns = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    try:
        load_env_file(Path(ns.env_file))
        detected_tokens = ensure_mineru_token()
        runner = resolve_runner(project_root, ns.python_exe, ns.use_uv)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    config_path = Path(ns.config_path) if ns.config_path else None
    if not config_path and not ns.no_auto_config:
        config_path = build_auto_config(project_root)
        print(f"Auto config: {config_path}")

    common_args: list[str] = []
    if config_path:
        common_args.extend(["--config-path", str(config_path)])

    print("== MinerU UV Test ==")
    print(f"Project: {project_root}")
    print(f"Env file: {ns.env_file}")
    print(f"Detected token count: {detected_tokens}")
    print(f"Limit: {ns.limit}")
    print(f"Runner: {' '.join(runner)}")

    try:
        run_cli(
            project_root,
            runner,
            ["doctor", "--check-local-db-lock", "--check-mineru-config", *common_args],
        )

        if not ns.skip_update:
            update_args = ["update-db", "--fulltext", "--limit", str(ns.limit)]
            if ns.force_rebuild:
                update_args.append("--force-rebuild")
            if ns.db_path:
                update_args.extend(["--db-path", ns.db_path])
            update_args.extend(common_args)
            run_cli(project_root, runner, update_args)

        run_cli(project_root, runner, ["db-status", *common_args])
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
