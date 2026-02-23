#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="$PROJECT_ROOT/.env"
LIMIT="20"
FORCE_REBUILD="false"
SKIP_UPDATE="false"
CONFIG_PATH=""
DB_PATH=""
USE_UV="false"
AUTO_PROJECT_CONFIG="true"

usage() {
  cat <<'USAGE'
Usage: scripts/test_mineru_uv.sh [options]

Options:
  --env-file PATH        Path to .env file (default: <project>/.env)
  --limit N              Limit items for update-db test (default: 20)
  --force-rebuild        Add --force-rebuild to update-db
  --skip-update          Only run doctor + db-status, skip update-db
  --config-path PATH     Pass --config-path to CLI commands
  --db-path PATH         Pass --db-path to update-db
  --no-auto-config       Disable auto-generated project test config
  --use-uv               Force using 'uv run python -m zotero_mcp.cli'
  --python PATH          Python executable to run cli module (default: .venv/bin/python)
  -h, --help             Show help

Environment variables in .env (any one is accepted):
  MINERU_TOKENS          Comma-separated tokens
  MINERU_TOKEN           Single token
  MINERU_API_TOKEN       Single token alias (auto-mapped to MINERU_TOKEN)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --force-rebuild)
      FORCE_REBUILD="true"
      shift
      ;;
    --skip-update)
      SKIP_UPDATE="true"
      shift
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --db-path)
      DB_PATH="$2"
      shift 2
      ;;
    --no-auto-config)
      AUTO_PROJECT_CONFIG="false"
      shift
      ;;
    --use-uv)
      USE_UV="true"
      shift
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: .env file not found at: $ENV_FILE" >&2
  exit 1
fi

if [[ -z "${PYTHON_EXE:-}" ]]; then
  PYTHON_EXE="$PROJECT_ROOT/.venv/bin/python"
fi

if [[ "$USE_UV" == "true" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: --use-uv specified but uv is not installed or not in PATH." >&2
    exit 1
  fi
  RUNNER=("uv" "run" "$PYTHON_EXE" "-m" "zotero_mcp.cli")
else
  if [[ -x "$PYTHON_EXE" ]]; then
    RUNNER=("$PYTHON_EXE" "-m" "zotero_mcp.cli")
  elif command -v uv >/dev/null 2>&1; then
    RUNNER=("uv" "run" "$PYTHON_EXE" "-m" "zotero_mcp.cli")
  else
    echo "Error: python executable not found: $PYTHON_EXE" >&2
    exit 1
  fi
fi

# Load .env into current shell.
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

# Backward-compatible alias support.
if [[ -n "${MINERU_API_TOKEN:-}" && -z "${MINERU_TOKEN:-}" && -z "${MINERU_TOKENS:-}" ]]; then
  export MINERU_TOKEN="$MINERU_API_TOKEN"
fi

if [[ -z "${MINERU_TOKEN:-}" && -z "${MINERU_TOKENS:-}" ]]; then
  echo "Error: no MinerU token found in $ENV_FILE." >&2
  echo "Please set MINERU_TOKEN, MINERU_TOKENS, or MINERU_API_TOKEN." >&2
  exit 1
fi

if [[ -n "${MINERU_TOKENS:-}" ]]; then
  TOKEN_COUNT=$(python3 - <<'PY'
import os
s = os.getenv("MINERU_TOKENS", "")
print(len([t for t in [x.strip() for x in s.split(",")] if t]))
PY
)
else
  TOKEN_COUNT="1"
fi

echo "== MinerU UV Test =="
echo "Project: $PROJECT_ROOT"
echo "Env file: $ENV_FILE"
echo "Detected token count: $TOKEN_COUNT"
echo "Limit: $LIMIT"
echo "Runner: ${RUNNER[*]}"

if [[ -z "$CONFIG_PATH" && "$AUTO_PROJECT_CONFIG" == "true" ]]; then
  mkdir -p "$PROJECT_ROOT/.tmp"
  CONFIG_PATH="$PROJECT_ROOT/.tmp/mineru-test-config.json"
  LOCATOR_DB_PATH="$PROJECT_ROOT/.tmp/locator.db"
  MD_STORE_DIR="$PROJECT_ROOT/.tmp/md_store"
  CHROMA_DB_DIR="$PROJECT_ROOT/.tmp/chroma_db"
  cat > "$CONFIG_PATH" <<JSON
{
  "semantic_search": {
    "embedding_model": "default",
    "collection_name": "zotero_library_chunks_v2",
    "persist_directory": "$CHROMA_DB_DIR",
    "extraction_mode": "mineru",
    "mineru": {
      "enabled": true
    },
    "locator_db": {
      "path": "$LOCATOR_DB_PATH"
    },
    "md_store": {
      "base_dir": "$MD_STORE_DIR"
    }
  }
}
JSON
  echo "Auto config: $CONFIG_PATH"
fi

COMMON_ARGS=()
if [[ -n "$CONFIG_PATH" ]]; then
  COMMON_ARGS+=("--config-path" "$CONFIG_PATH")
fi

UPDATE_ARGS=("--fulltext" "--limit" "$LIMIT")
if [[ "$FORCE_REBUILD" == "true" ]]; then
  UPDATE_ARGS+=("--force-rebuild")
fi
if [[ -n "$DB_PATH" ]]; then
  UPDATE_ARGS+=("--db-path" "$DB_PATH")
fi

echo
printf 'Running: %q ' env "PYTHONPATH=$PROJECT_ROOT/src" "${RUNNER[@]}" doctor --check-local-db-lock --check-mineru-config ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}
echo
env "PYTHONPATH=$PROJECT_ROOT/src" "${RUNNER[@]}" doctor --check-local-db-lock --check-mineru-config ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}

if [[ "$SKIP_UPDATE" != "true" ]]; then
  echo
  printf 'Running: %q ' env "PYTHONPATH=$PROJECT_ROOT/src" "${RUNNER[@]}" update-db "${UPDATE_ARGS[@]}" ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}
  echo
  env "PYTHONPATH=$PROJECT_ROOT/src" "${RUNNER[@]}" update-db "${UPDATE_ARGS[@]}" ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}
fi

echo
printf 'Running: %q ' env "PYTHONPATH=$PROJECT_ROOT/src" "${RUNNER[@]}" db-status ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}
echo
env "PYTHONPATH=$PROJECT_ROOT/src" "${RUNNER[@]}" db-status ${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}

echo
echo "Done."
