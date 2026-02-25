#!/usr/bin/env python3
"""Local integration runner for MinerU vectorization flow.

This script replaces `scripts/test_mineru_uv.sh` with a Python equivalent.
It is intended for manual local verification, not CI execution.

Usage examples:
  - Full flow (doctor + update-db --fulltext + db-status, default limit=2):
      .venv/bin/python tests/test_mineru_uv.py
  - With explicit zotero.sqlite path:
      .venv/bin/python tests/test_mineru_uv.py --db-path "$HOME/Zotero/zotero.sqlite"
  - Config checks only (skip update-db):
      .venv/bin/python tests/test_mineru_uv.py --skip-update

Behavior:
  - Reads MinerU token(s) from project `.env`.
  - Writes isolated runtime artifacts under `.tmp/mineru-home-<timestamp>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
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
        default=2,
        type=int,
        help="Limit items for update-db test (default: 2)",
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
        if python_exe:
            return ["uv", "run", python_exe, "-m", "zotero_mcp.cli"]
        venv_python = project_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            return ["uv", "run", str(venv_python), "-m", "zotero_mcp.cli"]
        return ["uv", "run", "python", "-m", "zotero_mcp.cli"]

    if Path(resolved_python).exists():
        return [resolved_python, "-m", "zotero_mcp.cli"]
    if uv_bin:
        return ["uv", "run", "python", "-m", "zotero_mcp.cli"]
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


def build_auto_config(project_root: Path, real_home: Path) -> tuple[Path, dict[str, str]]:
    tmp_dir = project_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_home = tmp_dir / f"mineru-home-{run_id}"
    config_dir = run_home / ".config" / "zotero-mcp"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "config.json"

    default_db_path = real_home / "Zotero" / "zotero.sqlite"
    zotero_db_path = str(default_db_path) if default_db_path.exists() else ""

    config = {
        "semantic_search": {
            "embedding_model": "default",
            "collection_name": "zotero_library_chunks_v2",
            "extraction_mode": "mineru",
            "mineru": {"enabled": True},
            "zotero_db_path": zotero_db_path,
        }
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    env_overrides = {
        "HOME": str(run_home),
        "XDG_CONFIG_HOME": str(run_home / ".config"),
        # Keep model caches on the real user cache to avoid re-downloading in isolated HOME.
        "HF_HOME": str(real_home / ".cache" / "huggingface"),
        "TRANSFORMERS_CACHE": str(real_home / ".cache" / "huggingface" / "hub"),
        "SENTENCE_TRANSFORMERS_HOME": str(real_home / ".cache" / "torch" / "sentence_transformers"),
    }
    return config_path, env_overrides


def run_cli(
    project_root: Path,
    runner: list[str],
    args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> None:
    cmd = runner + args
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    if env_overrides:
        env.update(env_overrides)
    mode_hint = " (isolated HOME)" if env_overrides else ""
    printable = " ".join(shlex.quote(x) for x in cmd)
    print("", flush=True)
    print(f"Running{mode_hint}: {printable}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    ns = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    real_home = Path.home()
    os.chdir(project_root)

    try:
        load_env_file(Path(ns.env_file))
        detected_tokens = ensure_mineru_token()
        runner = resolve_runner(project_root, ns.python_exe, ns.use_uv)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    config_path = Path(ns.config_path) if ns.config_path else None
    isolated_env: dict[str, str] = {}
    if not config_path and not ns.no_auto_config:
        config_path, isolated_env = build_auto_config(project_root, real_home)
        print(f"Auto config: {config_path}", flush=True)

    common_args: list[str] = []
    # Auto-config simulates real default path under isolated HOME, so no --config-path needed.
    if config_path and not isolated_env:
        common_args.extend(["--config-path", str(config_path)])

    print("== MinerU UV Test ==", flush=True)
    print(f"Project: {project_root}", flush=True)
    print(f"Env file: {ns.env_file}", flush=True)
    print(f"Detected token count: {detected_tokens}", flush=True)
    print(f"Limit: {ns.limit}", flush=True)
    print(f"Runner: {' '.join(runner)}", flush=True)
    if isolated_env and not ns.db_path:
        cfg_json = json.loads(config_path.read_text(encoding="utf-8"))
        if not cfg_json.get("semantic_search", {}).get("zotero_db_path"):
            print(
                "Warning: auto config could not detect zotero.sqlite; pass --db-path for update-db.",
                flush=True,
            )

    try:
        # Keep local DB lock check on real HOME for realism and stability.
        run_cli(
            project_root,
            runner,
            ["doctor", "--check-local-db-lock"],
        )
        run_cli(
            project_root,
            runner,
            ["doctor", "--check-mineru-config", *common_args],
            env_overrides=isolated_env or None,
        )

        if not ns.skip_update:
            update_args = ["update-db", "--fulltext", "--limit", str(ns.limit)]
            if ns.force_rebuild:
                update_args.append("--force-rebuild")
            if ns.db_path:
                update_args.extend(["--db-path", ns.db_path])
            update_args.extend(common_args)
            run_cli(project_root, runner, update_args, env_overrides=isolated_env or None)

        run_cli(project_root, runner, ["db-status", *common_args], env_overrides=isolated_env or None)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except OSError as exc:
        print(f"Failed to execute command: {exc}", file=sys.stderr)
        return 1

    print("", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
