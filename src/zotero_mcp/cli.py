"""
Command-line interface for Zotero MCP server.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# NOTE: Do NOT import zotero_mcp.server at module level.
# That triggers heavy imports (FastMCP, ChromaDB, sentence-transformers, torch)
# which take several seconds. Import lazily only when needed (serve command).
# This allows CLI commands like update-db to print "Starting up..." instantly.


def obfuscate_sensitive_value(value, keep_chars=4):
    """Obfuscate sensitive values by showing only the first few characters."""
    if not value or not isinstance(value, str):
        return value
    if len(value) <= keep_chars:
        return "*" * len(value)
    return value[:keep_chars] + "*" * (len(value) - keep_chars)


def obfuscate_config_for_display(config):
    """Create a copy of config with sensitive values obfuscated."""
    if not isinstance(config, dict):
        return config

    obfuscated = config.copy()
    sensitive_keys = [
        "ZOTERO_API_KEY",
        "ZOTERO_LIBRARY_ID",
        "ZOTERO_WEBDAV_URL",
        "ZOTERO_WEBDAV_USERNAME",
        "ZOTERO_WEBDAV_PASSWORD",
        "API_KEY",
        "LIBRARY_ID",
        "WEBDAV_URL",
        "WEBDAV_USERNAME",
        "WEBDAV_PASSWORD",
    ]

    for key in sensitive_keys:
        if key in obfuscated:
            obfuscated[key] = obfuscate_sensitive_value(obfuscated[key])

    return obfuscated


def load_claude_desktop_env_vars():
    """Load Zotero environment variables from Claude Desktop config unless globally disabled."""
    # Global guard to skip Claude detection entirely
    if str(os.environ.get("ZOTERO_NO_CLAUDE", "")).lower() in ("1", "true", "yes"):
        return {}
    from zotero_mcp.setup_helper import find_claude_config

    try:
        config_path = find_claude_config()
        if not config_path or not config_path.exists():
            return {}

        with open(config_path) as f:
            config = json.load(f)

        # Extract Zotero MCP server environment variables
        mcp_servers = config.get("mcpServers", {})
        zotero_config = mcp_servers.get("zotero", {})
        env_vars = zotero_config.get("env", {})

        return env_vars

    except Exception:
        return {}


def load_standalone_env_vars():
    """Load environment variables from standalone config (~/.config/zotero-mcp/config.json)."""
    try:
        from pathlib import Path
        cfg_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        if not cfg_path.exists():
            return {}
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("client_env", {}) or {}
    except Exception:
        return {}


def apply_environment_variables(env_vars):
    """Apply environment variables to current process."""
    for key, value in env_vars.items():
        if key not in os.environ:  # Don't override existing env vars
            os.environ[key] = str(value)


def _save_zotero_db_path_to_config(config_path: Path, db_path: str) -> None:
    """
    Save the Zotero database path to the configuration file.

    This allows users to specify --db-path once and have it remembered
    for subsequent runs without needing to specify it again.

    Args:
        config_path: Path to the configuration file
        db_path: Path to the Zotero database file
    """
    try:
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new one
        full_config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    full_config = json.load(f)
            except Exception:
                pass

        # Ensure semantic_search section exists
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        # Save the db_path
        full_config["semantic_search"]["zotero_db_path"] = db_path

        # Write back to file
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        # The config can hold credentials (API/embedding keys) — keep it
        # owner-only. Best-effort; no-op on platforms without POSIX perms.
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass

        print(f"Saved Zotero database path to config: {config_path}")

    except Exception as e:
        print(f"Warning: Could not save db_path to config: {e}")


def _semantic_config_path(path_arg: str | None) -> Path:
    return Path(path_arg) if path_arg else Path.home() / ".config" / "zotero-mcp" / "config.json"


def _warmup_reranker_in_background() -> None:
    """Preload the reranker (if enabled) off the request path — see issue #283.

    Runs in a daemon thread so server startup is never delayed and a failed or
    slow model load can never crash the server. No-op when the optional
    ``[semantic]`` extra isn't installed or the reranker is disabled.
    """
    import threading

    def _run() -> None:
        try:
            from zotero_mcp.semantic_search import warmup_reranker
        except Exception:
            return  # semantic extra not installed
        try:
            config_path = str(_semantic_config_path(None))
            if warmup_reranker(config_path):
                print("Reranker warmed up.", file=sys.stderr)
        except Exception:
            pass  # best-effort: never let warmup break serving

    threading.Thread(target=_run, daemon=True, name="zmcp-reranker-warmup").start()


def _print_update_stats(stats: dict) -> None:
    is_batch = stats.get("batch_mode") or stats.get("batch_submitted")
    batch_provider = stats.get("batch_provider", "openai")
    label = f"{_provider_label(batch_provider)} batch submission" if is_batch else "Database update"
    outcome = "failed" if stats.get("error") else "completed"
    print(f"\n{label} {outcome}:")
    print(f"- Total items: {stats.get('total_items', 0)}")
    print(f"- Processed: {stats.get('processed_items', 0)}")
    if stats.get("batch_submitted"):
        sub_count = stats.get('submitted_items', 0)
        tot_items = stats.get('total_items', 0)
        if sub_count != tot_items:
            print(f"- Submitted: {sub_count} chunks/records (from {tot_items} items)")
        else:
            print(f"- Submitted: {sub_count} records")
        print(f"- Estimated new records: {stats.get('estimated_added_items', 0)}")
        print(f"- Estimated existing records: {stats.get('estimated_updated_items', 0)}")
    else:
        print(f"- Added: {stats.get('added_items', 0)}")
        print(f"- Updated: {stats.get('updated_items', 0)}")
    print(f"- Skipped: {stats.get('skipped_items', 0)}")
    print(f"- Errors: {stats.get('errors', 0)}")
    print(f"- Duration: {stats.get('duration', 'Unknown')}")
    if stats.get("batch_submitted"):
        print(f"- Batch run: {stats.get('batch_run_id')}")
        print(f"- Manifest: {stats.get('batch_manifest')}")
        for batch_id in stats.get("batch_ids", []):
            print(f"- Batch ID: {batch_id}")
        if stats.get("batch_pending"):
            print(f"- Pending chunks (throttled): {stats['batch_pending']}")
        if stats.get("auto_loop"):
            al = stats["auto_loop"]
            print(
                f"- Auto-loop: {al.get('polls', 0)} polls, "
                f"{al.get('submitted_chunks', 0)} pending chunks submitted, "
                f"{al.get('imported_items', 0)} embeddings imported"
            )
            if al.get("stalled"):
                print(f"- WARNING: stalled chunks: {', '.join(al['stalled'])}")
        else:
            print("\nNext steps:")
            print("  zotero-mcp batch-status")
            print("  zotero-mcp batch-import")


def _provider_label(provider: str) -> str:
    """Human-readable label for a provider name (e.g. "openai" -> "OpenAI").

    Looks up the registered ``ProviderSpec.label`` so new batch-capable
    providers get sensible display names automatically; falls back to
    capitalizing the raw name for anything unregistered.
    """
    from zotero_mcp.embeddings.registry import PROVIDERS

    spec = PROVIDERS.get(provider)
    return spec.label if spec else provider.capitalize()


def _detect_batch_provider(search) -> str:
    """Infer the batch provider from the configured embedding model."""
    from zotero_mcp.embeddings.registry import batch_capable_providers

    model = search.chroma_client.embedding_model
    providers = batch_capable_providers()
    if model in providers:
        return model
    choices = " or ".join(f"--provider {p}" for p in providers)
    raise ValueError(
        f"Configured embedding model '{model}' has no Batch API support; "
        f"pass {choices} to select a manifest explicitly."
    )


def _batch_provider_choices() -> list[str]:
    """Providers with Batch API support, for ``--batch-provider``/``--provider``
    argparse choices.

    Importing the batch modules registers their adapters as a side effect
    (see the ``attach_batch_adapter`` calls at the bottom of
    ``openai_batch.py``/``gemini_batch.py``), which is what makes
    ``batch_capable_providers()`` non-empty. That import chain pulls in
    ``chromadb`` (via ``zotero_mcp.embeddings``), the one genuinely heavy
    dependency this module otherwise avoids at parse time — unavoidable here
    since argparse needs concrete ``choices=`` when the subparsers are built,
    which happens for every CLI invocation before command dispatch.
    """
    import zotero_mcp.gemini_batch  # noqa: F401 — import registers its batch adapter
    import zotero_mcp.openai_batch  # noqa: F401 — import registers its batch adapter
    from zotero_mcp.embeddings.registry import batch_capable_providers

    return batch_capable_providers()


def _print_batch_status(status: dict, provider: str = "openai") -> None:
    print(f"=== {_provider_label(provider)} Batch Status ===")
    print(f"Run: {status.get('run_id')}")
    print(f"Model: {status.get('model')}")
    print(f"Manifest: {status.get('manifest_path')}")
    print(f"Force rebuild: {status.get('force_full_rebuild', False)}")
    for batch in status.get("batches", []):
        counts = batch.get("request_counts") or {}
        if not isinstance(counts, dict):
            counts = {}
        print()
        print(f"Batch: {batch.get('batch_id') or '(not yet submitted)'}")
        print(f"- Status: {batch.get('status')}")
        print(f"- Requests: {batch.get('request_count', counts.get('total', 'Unknown'))}")
        if batch.get("request_tokens"):
            print(f"- Est. tokens: {batch['request_tokens']:,}")
        if counts:
            print(f"- Completed: {counts.get('completed', 0)}")
            print(f"- Failed: {counts.get('failed', 0)}")
        print(f"- Imported: {batch.get('imported_at') or 'No'}")
    batches = status.get("batches", [])
    pending = [b for b in batches if b.get("status") == "pending"]
    if pending:
        pending_tokens = sum(int(b.get("request_tokens") or 0) for b in pending)
        print(f"\nPending chunks (throttled): {len(pending)} (~{pending_tokens:,} tokens)")


def _print_batch_import(stats: dict, provider: str = "openai") -> None:
    print(f"=== {_provider_label(provider)} Batch Import ===")
    print(f"Run: {stats.get('run_id')}")
    print(f"Manifest: {stats.get('manifest_path')}")
    print(f"- Batches seen: {stats.get('batches_seen', 0)}")
    print(f"- Batches imported: {stats.get('batches_imported', 0)}")
    print(f"- Batches skipped: {stats.get('batches_skipped', 0)}")
    print(f"- Imported items: {stats.get('imported_items', 0)}")
    print(f"- Added: {stats.get('added_items', 0)}")
    print(f"- Updated: {stats.get('updated_items', 0)}")
    print(f"- Failed rows: {stats.get('failed_items', 0)}")
    print(f"- Missing rows: {stats.get('missing_items', 0)}")
    if stats.get("errors"):
        print("\nWarnings/errors:")
        for error in stats["errors"][:20]:
            print(f"- {error}")
        if len(stats["errors"]) > 20:
            print(f"- ... {len(stats['errors']) - 20} more")


def setup_zotero_environment():
    """Setup Zotero environment for CLI commands."""
    # Load standalone env first so global flags (e.g., ZOTERO_NO_CLAUDE) take effect
    standalone_env_vars = load_standalone_env_vars()
    apply_environment_variables(standalone_env_vars)

    # Respect global switch to disable Claude detection
    no_claude = str(os.environ.get("ZOTERO_NO_CLAUDE", "")).lower() in ("1", "true", "yes")

    # Load and apply Claude Desktop env unless disabled
    if not no_claude:
        claude_env_vars = load_claude_desktop_env_vars()
        apply_environment_variables(claude_env_vars)

    # Apply fallback defaults for local Zotero if no config found.
    # Only apply when no API key is configured — if an API key exists,
    # the user intends web API mode and we should not force local mode.
    if not os.environ.get("ZOTERO_API_KEY"):
        fallback_env_vars = {
            "ZOTERO_LOCAL": "true",
            "ZOTERO_LIBRARY_ID": "0",
        }
        apply_environment_variables(fallback_env_vars)


def _normalize_help_args(argv: list[str]) -> list[str]:
    """Support `zotero-mcp help [command]` in addition to argparse's `--help`."""
    if not argv or argv[0] != "help":
        return argv
    if len(argv) == 1:
        return ["--help"]
    return [*argv[1:], "--help"]


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Zotero Model Context Protocol server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Batch API indexing (OpenAI or Gemini):\n"
            "  zotero-mcp update-db --openai-batch     Submit OpenAI embeddings asynchronously\n"
            "  zotero-mcp update-db --gemini-batch     Submit Gemini embeddings asynchronously\n"
            "  zotero-mcp update-db --gemini-batch --auto-loop\n"
            "                                          Submit throttled, then poll/import/submit to completion\n"
            "  zotero-mcp batch-status                 Check submitted batch status\n"
            "  zotero-mcp batch-import                 Import completed embeddings\n"
            "  zotero-mcp help update-db               Show update-db options\n"
        ),
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command (default behavior)
    server_parser = subparsers.add_parser("serve", help="Run the MCP server")
    server_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    server_parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to for SSE transport (default: localhost)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to for SSE transport (default: 8000)",
    )

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Configure zotero-mcp (Claude Desktop or standalone)")
    setup_parser.add_argument("--no-local", action="store_true",
                             help="Configure for Zotero Web API instead of local API")
    setup_parser.add_argument("--api-key", help="Zotero API key (only needed with --no-local)")
    setup_parser.add_argument("--library-id", help="Zotero library ID (only needed with --no-local)")
    setup_parser.add_argument("--library-type", choices=["user", "group"], default="user",
                             help="Zotero library type (only needed with --no-local)")
    setup_parser.add_argument("--no-claude", action="store_true",
                             help="Skip Claude Desktop config; write standalone config for web-based clients")
    setup_parser.add_argument("--config-path", help="Path to Claude Desktop config file")
    setup_parser.add_argument("--skip-semantic-search", action="store_true",
                             help="Skip semantic search configuration")
    setup_parser.add_argument("--semantic-config-only", action="store_true",
                             help="Only configure semantic search, skip Zotero setup")

    # Update database command
    update_db_parser = subparsers.add_parser("update-db", help="Update semantic search database")
    update_db_parser.add_argument("--force-rebuild", action="store_true",
                                 help="Force complete rebuild of the database")
    update_db_parser.add_argument("--limit", type=int,
                                 help="Limit number of items to process (for testing)")
    update_db_parser.add_argument("--fulltext", action="store_true",
                                 help="Extract fulltext content from local Zotero database (slower but more comprehensive)")
    update_db_parser.add_argument("--config-path",
                                 help="Path to semantic search configuration file")
    update_db_parser.add_argument("--db-path",
                                 help="Path to Zotero database file (zotero.sqlite), overrides config")
    update_db_parser.add_argument("--extraction-workers", type=int, default=None,
                                 help="Concurrent PDF extractions during --fulltext (default: config "
                                      "'semantic_search.extraction.workers' or 1; capped at CPU count / 8)")
    update_db_parser.add_argument("--clear-fulltext-cache", action="store_true",
                                 help="Clear the transient extracted-fulltext cache before running")
    update_db_parser.add_argument("-v", "--verbose", action="store_true",
                                 help="Enable verbose output including real-time API load and latency telemetry")

    openai_batch_group = update_db_parser.add_mutually_exclusive_group()
    openai_batch_group.add_argument("--openai-batch", dest="openai_batch", action="store_true",
                                   help="Submit OpenAI embeddings through the asynchronous Batch API "
                                        "(deprecated: use --batch [--batch-provider openai])")
    openai_batch_group.add_argument("--no-openai-batch", dest="openai_batch", action="store_false",
                                   help="Use realtime embeddings even if OpenAI Batch API is enabled in config "
                                        "(deprecated: use --no-batch)")
    update_db_parser.set_defaults(openai_batch=None)
    gemini_batch_group = update_db_parser.add_mutually_exclusive_group()
    gemini_batch_group.add_argument("--gemini-batch", dest="gemini_batch", action="store_true",
                                   help="Submit Gemini embeddings through the asynchronous Batch API (experimental) "
                                        "(deprecated: use --batch [--batch-provider gemini])")
    gemini_batch_group.add_argument("--no-gemini-batch", dest="gemini_batch", action="store_false",
                                   help="Use realtime embeddings even if Gemini Batch API is enabled in config "
                                        "(deprecated: use --no-batch)")
    update_db_parser.set_defaults(gemini_batch=None)
    batch_group = update_db_parser.add_mutually_exclusive_group()
    batch_group.add_argument("--batch", dest="use_batch", action="store_true",
                            help="Submit embeddings through the asynchronous Batch API for the "
                                 "configured embedding model (generic form of --openai-batch/--gemini-batch; "
                                 "pair with --batch-provider to select a provider explicitly)")
    batch_group.add_argument("--no-batch", dest="use_batch", action="store_false",
                            help="Use realtime embeddings even if Batch API is enabled in config")
    update_db_parser.set_defaults(use_batch=None)
    update_db_parser.add_argument("--batch-provider", choices=_batch_provider_choices(), default=None,
                                 help="Batch provider to use with --batch (default: the configured "
                                      "embedding model, if it supports the Batch API)")
    update_db_parser.add_argument("--batch-max-tokens", type=int, default=None,
                                 help="Max estimated tokens enqueued with the batch provider at once "
                                      "(overrides semantic_search.<provider>_batch.batch_max_enqueued_tokens). "
                                      "Tier presets — Gemini/OpenAI: Tier 1 = 450000/2500000, "
                                      "Tier 2 = 4500000/18000000, Tier 3 = 9000000/90000000")
    update_db_parser.add_argument("--batch-max-requests", type=int, default=None,
                                 help="Max requests per uploaded batch JSONL file (default 50000)")
    update_db_parser.add_argument("--auto-loop", action="store_true",
                                 help="After submitting, poll status, import completed batches, and "
                                      "submit pending chunks until the whole run is imported "
                                      "(requires batch mode)")
    update_db_parser.add_argument("--batch-poll-interval", type=int, default=60,
                                 help="Seconds between --auto-loop status polls (default: 60)")

    # Batch lifecycle commands (provider-neutral). The openai-batch-* forms
    # are kept as backward-compatible aliases pinned to --provider openai.
    batch_status_parser = subparsers.add_parser("batch-status", help="Show embedding Batch API status")
    batch_status_parser.add_argument("--provider", choices=_batch_provider_choices(), default=None,
                                     help="Batch provider (default: auto-detect from configured embedding model)")
    batch_status_parser.add_argument("--batch-id", action="append",
                                     help="Specific batch ID to inspect; can be repeated")
    batch_status_parser.add_argument("--config-path",
                                     help="Path to semantic search configuration file")

    batch_import_parser = subparsers.add_parser("batch-import", help="Import completed batch embeddings")
    batch_import_parser.add_argument("--provider", choices=_batch_provider_choices(), default=None,
                                     help="Batch provider (default: auto-detect from configured embedding model)")
    batch_import_parser.add_argument("--batch-id", action="append",
                                     help="Specific batch ID to import; can be repeated")
    batch_import_parser.add_argument("--config-path",
                                     help="Path to semantic search configuration file")

    # Backward-compatible OpenAI-specific aliases
    openai_status_parser = subparsers.add_parser("openai-batch-status", help="Show OpenAI Batch API status")
    openai_status_parser.add_argument("--batch-id", action="append",
                                      help="Specific OpenAI batch ID to inspect; can be repeated")
    openai_status_parser.add_argument("--config-path",
                                      help="Path to semantic search configuration file")

    openai_import_parser = subparsers.add_parser("openai-batch-import", help="Import completed OpenAI batch embeddings")
    openai_import_parser.add_argument("--batch-id", action="append",
                                      help="Specific OpenAI batch ID to import; can be repeated")
    openai_import_parser.add_argument("--config-path",
                                      help="Path to semantic search configuration file")

    # Database status command
    db_status_parser = subparsers.add_parser("db-status", help="Show semantic search database status")
    db_status_parser.add_argument("--config-path",
                                 help="Path to semantic search configuration file")

    # DB inspect command (sample and filter indexed docs; also supports stats)
    inspect_parser = subparsers.add_parser("db-inspect", help="Inspect indexed documents or show aggregate stats for the semantic DB")
    inspect_parser.add_argument("--limit", type=int, default=20, help="How many records to show (default: 20)")
    inspect_parser.add_argument("--filter", dest="filter_text", help="Substring to match in title or creators")
    inspect_parser.add_argument("--show-documents", action="store_true", help="Show beginning of stored document text")
    inspect_parser.add_argument("--stats", action="store_true", help="Show aggregate stats (formerly db-stats)")
    inspect_parser.add_argument("--config-path", help="Path to semantic search configuration file")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update zotero-mcp to the latest version")
    update_parser.add_argument("--check-only", action="store_true",
                              help="Only check for updates without installing")
    update_parser.add_argument("--force", action="store_true",
                              help="Force update even if already up to date")
    update_parser.add_argument("--method", choices=["pip", "uv", "conda", "pipx"],
                              help="Override auto-detected installation method")

    # Version command
    subparsers.add_parser("version", help="Print version information")

    # Setup info command
    subparsers.add_parser("setup-info", help="Show installation path and configuration info for MCP clients")

    args = parser.parse_args(_normalize_help_args(sys.argv[1:]))

    # If no command is provided, default to 'serve'
    if not args.command:
        args.command = "serve"
        # Also set default transport since we're defaulting to serve
        args.transport = "stdio"

    if args.command == "version":
        from zotero_mcp._version import __version__
        print(f"Zotero MCP v{__version__}")
        sys.exit(0)

    elif args.command == "setup-info":
        # Setup Zotero environment variables
        setup_zotero_environment()

        # Get the installation path
        executable_path = shutil.which("zotero-mcp")
        if not executable_path:
            executable_path = sys.executable + " -m zotero_mcp"

        # Determine whether Claude is disabled globally
        no_claude = str(os.environ.get("ZOTERO_NO_CLAUDE", "")).lower() in ("1", "true", "yes")

        # Load current environment configurations
        standalone_env_vars = load_standalone_env_vars()
        claude_env_vars = {} if no_claude else load_claude_desktop_env_vars()

        # Choose which env to display: prefer standalone if present or if Claude disabled
        display_env = standalone_env_vars if (no_claude or standalone_env_vars) else (claude_env_vars or {"ZOTERO_LOCAL": "true"})

        print("=== Zotero MCP Setup Information ===")
        print()
        print("🔧 Installation Details:")
        print(f"  Command path: {executable_path}")
        print(f"  Python path: {sys.executable}")

        # Detect installation method
        try:
            # Check if installed via uv
            result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, timeout=5)
            if "zotero-mcp-server" in result.stdout or "zotero-mcp" in result.stdout:
                print("  Installation method: uv tool")
            else:
                # Check pip
                result = subprocess.run([sys.executable, "-m", "pip", "show", "zotero-mcp-server"],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("  Installation method: pip")
                else:
                    print("  Installation method: unknown")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            print("  Installation method: unknown")

        print()
        print("⚙️  MCP Client Configuration:")
        print(f"  Command: {executable_path}")
        print("  Arguments: [] (empty)")

        # Show environment variables with obfuscated sensitive values
        obfuscated_env_vars = obfuscate_config_for_display(display_env)
        print(f"  Environment (single-line): {json.dumps(obfuscated_env_vars, separators=(',', ':'))}")
        print("  💡 Note: This shows client config. Shell variables may override for CLI use.")
        print(f"  Claude integration: {'disabled' if no_claude else 'enabled'}")

        # Only show Claude Desktop config if not globally disabled
        if not no_claude:
            print()
            print("For Claude Desktop (claude_desktop_config.json):")
            config_snippet = {
                "mcpServers": {
                    "zotero": {
                        "command": executable_path,
                        "env": obfuscated_env_vars
                    }
                }
            }
            print(json.dumps(config_snippet, indent=2))

        # Show semantic search database info with detailed statistics
        print()
        print("🧠 Semantic Search Database:")

        # Check for semantic search config
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        if config_path.exists():
            try:
                from zotero_mcp.semantic_search import create_semantic_search

                # Get database status (similar to db-status command)
                search = create_semantic_search(str(config_path))
                status = search.get_database_status()

                collection_info = status.get("collection_info", {})

                print("  Status: ✅ Configuration file found")
                print(f"  Config path: {config_path}")
                print(f"  Collection: {collection_info.get('name', 'Unknown')}")
                print(f"  Document count: {collection_info.get('count', 0)}")
                print(f"  Embedding model: {collection_info.get('embedding_model', 'Unknown')}")
                print(f"  Database path: {collection_info.get('persist_directory', 'Unknown')}")

                update_config = status.get("update_config", {})
                batch_config = status.get("openai_batch", {})
                print(f"  Auto update: {update_config.get('auto_update', False)}")
                print(f"  Update frequency: {update_config.get('update_frequency', 'manual')}")
                print(f"  Last update: {update_config.get('last_update', 'Never')}")
                print(f"  Should update: {status.get('should_update', False)}")
                print(f"  OpenAI Batch API: {'active' if batch_config.get('active') else 'inactive'}")

                if collection_info.get('error'):
                    print(f"  Error: {collection_info['error']}")

            except Exception as e:
                print("  Status: ⚠️ Configuration found but database error")
                print(f"  Error: {e}")
        else:
            print("  Status: ⚠️ Not configured")
            print("  💡 Run 'zotero-mcp setup' to configure semantic search")

        sys.exit(0)

    elif args.command == "setup":
        from zotero_mcp.setup_helper import main as setup_main
        sys.exit(setup_main(args))

    elif args.command == "update-db":
        # argparse mutually-exclusive groups can't cross-check between
        # groups, so the generic --batch/--no-batch vs. legacy
        # --openai-batch/--gemini-batch conflict is caught here instead —
        # before any Zotero/ChromaDB setup, since it only depends on parsed
        # args.
        if args.use_batch is not None and (args.openai_batch is not None or args.gemini_batch is not None):
            print(
                "Error: --batch/--no-batch cannot be combined with "
                "--openai-batch/--no-openai-batch or --gemini-batch/--no-gemini-batch. "
                "Use --batch [--batch-provider openai|gemini] instead.",
                file=sys.stderr,
            )
            sys.exit(1)

        import logging
        log_level = logging.INFO if getattr(args, "verbose", False) else logging.WARNING
        logging.basicConfig(level=log_level, format="%(message)s", force=True)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)



        # Setup Zotero environment variables
        setup_zotero_environment()


        from zotero_mcp.semantic_search import create_semantic_search

        # Determine config path
        config_path = _semantic_config_path(args.config_path)

        print(f"Using configuration: {config_path}")

        # Get optional db_path override from CLI
        db_path = getattr(args, 'db_path', None)
        if db_path:
            print(f"Using custom Zotero database: {db_path}")
            # Save the db_path to config file for future use
            _save_zotero_db_path_to_config(config_path, db_path)

        try:
            # Create semantic search instance with optional db_path override
            search = create_semantic_search(str(config_path), db_path=db_path)
            if args.openai_batch is True and search.chroma_client.embedding_model != "openai":
                print("Error: --openai-batch requires ZOTERO_EMBEDDING_MODEL=openai", file=sys.stderr)
                sys.exit(1)
            if args.gemini_batch is True and search.chroma_client.embedding_model != "gemini":
                print("Error: --gemini-batch requires ZOTERO_EMBEDDING_MODEL=gemini", file=sys.stderr)
                sys.exit(1)

            if args.clear_fulltext_cache:
                from zotero_mcp import fulltext_cache
                cleared = fulltext_cache.clear_all(config_path=str(config_path))
                print(f"Cleared {cleared} cached fulltext entr{'y' if cleared == 1 else 'ies'}.")

            if args.auto_loop:
                batch_requested = (
                    args.use_batch is True
                    or args.batch_provider is not None
                    or args.openai_batch is True
                    or args.gemini_batch is True
                    or search._resolve_openai_batch_enabled(args.openai_batch)
                    or search._resolve_gemini_batch_enabled(args.gemini_batch)
                )
                if not batch_requested:
                    print(
                        "Error: --auto-loop requires batch mode "
                        "(--batch/--openai-batch/--gemini-batch or batch enabled in config).",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            print("Starting database update...")
            if args.fulltext:
                from zotero_mcp.utils import is_local_mode
                if not is_local_mode():
                    print(
                        "Error: --fulltext requires local mode but ZOTERO_LOCAL is not enabled.\n"
                        "Full-text indexing needs access to Zotero's local database.\n"
                        "Set ZOTERO_LOCAL=true or run 'zotero-mcp setup' to enable local mode.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print("Extracting full-text content from local Zotero database...")
            stats = search.update_database(
                force_full_rebuild=args.force_rebuild,
                limit=args.limit,
                extract_fulltext=args.fulltext,
                use_openai_batch=args.openai_batch,
                use_gemini_batch=args.gemini_batch,
                use_batch=args.use_batch,
                batch_provider=args.batch_provider,
                extraction_workers=args.extraction_workers,
                batch_max_tokens=args.batch_max_tokens,
                batch_max_requests=args.batch_max_requests,
                auto_loop=args.auto_loop,
                batch_poll_interval=args.batch_poll_interval,
            )

            _print_update_stats(stats)

            if stats.get('error'):
                print(f"Error: {stats['error']}")
                sys.exit(1)

        except Exception as e:
            print(f"Error updating database: {e}")
            sys.exit(1)

    elif args.command in ("batch-status", "openai-batch-status"):
        setup_zotero_environment()

        from zotero_mcp.semantic_search import create_semantic_search

        config_path = _semantic_config_path(args.config_path)
        provider = "openai" if args.command == "openai-batch-status" else getattr(args, "provider", None)
        try:
            search = create_semantic_search(str(config_path))
            provider = provider or _detect_batch_provider(search)
            if provider == "gemini":
                status = search.get_gemini_batch_status(batch_ids=args.batch_id)
            else:
                status = search.get_openai_batch_status(batch_ids=args.batch_id)
            _print_batch_status(status, provider)
        except Exception as e:
            print(f"Error getting {_provider_label(provider) if provider else 'embedding'} batch status: {e}")
            sys.exit(1)

    elif args.command in ("batch-import", "openai-batch-import"):
        setup_zotero_environment()

        from zotero_mcp.semantic_search import create_semantic_search

        config_path = _semantic_config_path(args.config_path)
        provider = "openai" if args.command == "openai-batch-import" else getattr(args, "provider", None)
        try:
            search = create_semantic_search(str(config_path))
            provider = provider or _detect_batch_provider(search)
            if provider == "gemini":
                stats = search.import_gemini_batch(batch_ids=args.batch_id)
            else:
                stats = search.import_openai_batch(batch_ids=args.batch_id)
            _print_batch_import(stats, provider)
        except Exception as e:
            print(f"Error importing {_provider_label(provider) if provider else 'embedding'} batch: {e}")
            sys.exit(1)

    elif args.command == "db-status":
        # Setup Zotero environment variables
        setup_zotero_environment()

        from zotero_mcp.semantic_search import create_semantic_search

        # Determine config path
        config_path = args.config_path
        if not config_path:
            config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        else:
            config_path = Path(config_path)

        try:
            print("Connecting to ChromaDB database...", flush=True)
            # Create semantic search instance
            search = create_semantic_search(str(config_path))

            print("Fetching collection status...", flush=True)
            # Get database status
            status = search.get_database_status()

            print("=== Semantic Search Database Status ===")

            collection_info = status.get("collection_info", {})
            print(f"Collection: {collection_info.get('name', 'Unknown')}")
            print(f"Document count: {collection_info.get('count', 0)}")
            print(f"Embedding model: {collection_info.get('embedding_model', 'Unknown')}")
            print(f"Database path: {collection_info.get('persist_directory', 'Unknown')}")

            update_config = status.get("update_config", {})
            batch_config = status.get("openai_batch", {})
            print("\nUpdate configuration:")
            print(f"- Auto update: {update_config.get('auto_update', False)}")
            print(f"- Frequency: {update_config.get('update_frequency', 'manual')}")
            print(f"- Last update: {update_config.get('last_update', 'Never')}")
            print(f"- Should update: {status.get('should_update', False)}")
            print(f"- OpenAI Batch API: {'active' if batch_config.get('active') else 'inactive'}")

            if collection_info.get('error'):
                print(f"\nError: {collection_info['error']}")

        except Exception as e:
            print(f"Error getting database status: {e}")
            sys.exit(1)

    elif args.command == "db-inspect":
        # Setup Zotero environment variables
        setup_zotero_environment()

        from collections import Counter

        from zotero_mcp.semantic_search import create_semantic_search

        # Determine config path
        config_path = args.config_path
        if not config_path:
            config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        else:
            config_path = Path(config_path)

        try:
            search = create_semantic_search(str(config_path))
            client = search.chroma_client
            col = client.collection

            if args.stats:
                # Show aggregate stats (merged from former db-stats)
                meta = col.get(include=["metadatas"])  # type: ignore
                metas = meta.get("metadatas", [])
                print("=== Semantic DB Inspection (Stats) ===")
                info = client.get_collection_info()
                print(f"Collection: {info.get('name')} @ {info.get('persist_directory')}")
                print(f"Count: {info.get('count')}")

                # Item type distribution
                item_types = [ (m or {}).get("item_type", "") for m in metas ]
                ct_types = Counter(item_types)
                print("Item types:")
                for t, c in ct_types.most_common(20):
                    print(f"  {t or '(missing)'}: {c}")

                # Fulltext coverage by type (pdf/html)
                coverage = {}
                for m in metas:
                    m = m or {}
                    t = m.get("item_type", "") or "(missing)"
                    cov = coverage.setdefault(t, {"total": 0, "with_fulltext": 0, "pdf": 0, "html": 0})
                    cov["total"] += 1
                    if m.get("has_fulltext"):
                        cov["with_fulltext"] += 1
                        src = (m.get("fulltext_source") or "").lower()
                        if src == "pdf":
                            cov["pdf"] += 1
                        elif src == "html":
                            cov["html"] += 1
                print("Fulltext coverage (by type):")
                for t, cov in coverage.items():
                    print(f"  {t}: {cov['with_fulltext']}/{cov['total']} (pdf:{cov['pdf']}, html:{cov['html']})")

                # Common titles (may indicate duplicates)
                titles = [ (m or {}).get("title", "") for m in metas ]
                from collections import Counter as _Counter
                ct_titles = _Counter([t for t in titles if t])
                common = [(t,c) for t,c in ct_titles.most_common(10)]
                if common:
                    print("Common titles:")
                    for t, c in common:
                        print(f"  {t[:80]}{'...' if len(t)>80 else ''}: {c}")
                return

            include = ["metadatas"]
            if args.show_documents:
                include.append("documents")

            # Fetch up to limit; filter client-side if requested
            data = col.get(limit=args.limit, include=include)

            print("=== Semantic DB Inspection ===")
            total = client.get_collection_info().get("count", 0)
            print(f"Total documents: {total}")
            print(f"Showing up to: {args.limit}")

            shown = 0
            for i, meta in enumerate(data.get("metadatas", [])):
                meta = meta or {}
                title = meta.get("title", "")
                creators = meta.get("creators", "")
                if args.filter_text:
                    needle = args.filter_text.lower()
                    if needle not in (title or "").lower() and needle not in (creators or "").lower():
                        continue
                print(f"- {title} | {creators}")
                if args.show_documents:
                    doc = (data.get("documents", [""])[i] or "").strip()
                    snippet = doc[:200].replace("\n", " ") + ("..." if len(doc) > 200 else "")
                    if snippet:
                        print(f"  doc: {snippet}")
                shown += 1
                if shown >= args.limit:
                    break

            if shown == 0:
                print("No records matched your filter.")

        except Exception as e:
            print(f"Error inspecting database: {e}")
            sys.exit(1)

    elif args.command == "update":
        from zotero_mcp.updater import update_zotero_mcp

        try:
            print("Checking for updates...")

            result = update_zotero_mcp(
                check_only=args.check_only,
                force=args.force,
                method=args.method
            )

            print("\n" + "="*50)
            print("UPDATE RESULTS")
            print("="*50)

            if args.check_only:
                print(f"Current version: {result.get('current_version', 'Unknown')}")
                print(f"Latest version: {result.get('latest_version', 'Unknown')}")
                print(f"Update needed: {result.get('needs_update', False)}")
                print(f"Status: {result.get('message', 'Unknown')}")
            else:
                if result.get('success'):
                    print("✅ Update completed successfully!")
                    print(f"Version: {result.get('current_version', 'Unknown')} → {result.get('latest_version', 'Unknown')}")
                    print(f"Method: {result.get('method', 'Unknown')}")
                    print(f"Message: {result.get('message', '')}")

                    print("\n📋 Next steps:")
                    print("• All configurations have been preserved")
                    print("• Restart Claude Desktop if it's running")
                    print("• Your semantic search database is intact")
                    print("• Run 'zotero-mcp version' to verify the update")
                else:
                    print("❌ Update failed!")
                    print(f"Error: {result.get('message', 'Unknown error')}")

                    if backup_dir := result.get('backup_dir'):
                        print(f"\n🔄 Backup created at: {backup_dir}")
                        print("You can manually restore configurations if needed")

                    sys.exit(1)

        except Exception as e:
            print(f"❌ Update error: {e}")
            sys.exit(1)

    elif args.command == "serve":
        # Lazy import — triggers heavy dependencies (FastMCP, ChromaDB, etc.)
        from zotero_mcp.server import mcp
        # Get transport with a default value if not specified
        transport = getattr(args, "transport", "stdio")
        # Ensure environment is initialized (Claude config or standalone config)
        setup_zotero_environment()
        # If the reranker is enabled, warm it up in the background so the first
        # semantic search doesn't pay the ~tens-of-seconds model load inside the
        # request path and time out (issue #283). Daemon thread: never blocks
        # startup, never crashes the server if loading fails.
        _warmup_reranker_in_background()
        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "streamable-http":
            host = getattr(args, "host", "localhost")
            port = getattr(args, "port", 8000)
            mcp.run(transport="streamable-http", host=host, port=port)
        elif transport == "sse":
            host = getattr(args, "host", "localhost")
            port = getattr(args, "port", 8000)
            import warnings
            warnings.warn("The SSE transport is deprecated and may be removed in a future version. New applications should use Streamable HTTP transport instead.", UserWarning)
            mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()
