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
    from zotero_mcp.setup_helper import find_existing_claude_configs

    try:
        # More than one Claude Desktop build can be installed (issue #392);
        # use the first config that actually configures the zotero server.
        for config_path in find_existing_claude_configs():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception:
                continue

            # Extract Zotero MCP server environment variables
            mcp_servers = config.get("mcpServers", {})
            zotero_config = mcp_servers.get("zotero", {})
            env_vars = zotero_config.get("env", {})
            if env_vars:
                return env_vars

        return {}

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

        # Save the db_path at the top level
        full_config["zotero_db_path"] = db_path

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


def _format_chunking_status(status: dict) -> str:
    """Describe what the next indexing run will actually do about chunking.

    Reports the effective behaviour, not the requested setting: the OpenAI
    Batch API path has no chunking step, so `chunking.enabled: true` there
    still yields one truncated vector per item (#416).
    """
    chunking = status.get("chunking", {})
    if not chunking.get("enabled"):
        return "disabled (item-level indexing)"
    if chunking.get("effective"):
        return "enabled"
    return "requested but NOT applied (Batch API path indexes item-level)"


# Providers with Batch API support, for argparse ``choices=``.
#
# Deliberately a literal rather than a call to
# ``embeddings.registry.batch_capable_providers()``: that list is only
# populated by importing openai_batch/gemini_batch, which pulls in chromadb
# (~330ms, measured) — and argparse needs concrete choices while the parser is
# being built, i.e. on *every* invocation including ``--version`` and
# ``--help``. Paying that to render a help string would undo #445, which made
# this module import nothing heavy at all. ``test_batch_provider_choices_match_registry``
# fails if this drifts from what is actually registered.
BATCH_PROVIDERS = ("openai", "gemini")


def _provider_label(provider: str) -> str:
    """Human-readable label for a provider name ("openai" -> "OpenAI").

    Reads the registered batch adapter's own label so a newly added provider
    displays correctly without touching this function. Only ever called from
    command handlers, by which point the registry is imported anyway, so the
    lazy import here costs nothing extra.
    """
    try:
        from zotero_mcp.embeddings.registry import PROVIDERS

        spec = PROVIDERS.get(provider)
        if spec is not None and spec.batch is not None:
            return spec.batch.label
    except Exception:
        pass
    return provider.capitalize()


def _print_update_stats(stats: dict) -> None:
    is_batch = stats.get("batch_mode") or stats.get("batch_submitted")
    batch_provider = stats.get("batch_provider", "openai")
    label = f"{_provider_label(batch_provider)} batch submission" if is_batch else "Database update"
    outcome = "failed" if stats.get("error") else "completed"
    print(f"\n{label} {outcome}:")
    print(f"- Total items: {stats.get('total_items', 0)}")
    print(f"- Processed: {stats.get('processed_items', 0)}")
    if stats.get("batch_submitted"):
        sub_count = stats.get("submitted_items", 0)
        tot_items = stats.get("total_items", 0)
        if sub_count != tot_items:
            print(f"- Submitted: {sub_count} chunks/records (from {tot_items} items)")
        else:
            print(f"- Submitted: {sub_count} records")
        print(f"- Estimated new items: {stats.get('estimated_added_items', 0)}")
        print(f"- Estimated existing items: {stats.get('estimated_updated_items', 0)}")
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
            loop = stats["auto_loop"]
            print(
                f"- Auto-loop: {loop.get('polls', 0)} polls, "
                f"{loop.get('submitted_chunks', 0)} pending chunks submitted, "
                f"{loop.get('imported_items', 0)} embeddings imported"
            )
            if loop.get("stalled"):
                print(f"- WARNING: stalled chunks: {', '.join(loop['stalled'])}")
        else:
            print("\nNext steps:")
            print("  zotero-mcp batch-status")
            print("  zotero-mcp batch-import")


def _detect_batch_provider(search) -> str:
    """Infer which provider's manifests to read from the configured model."""
    from zotero_mcp.embeddings.registry import batch_capable_providers

    model = search.chroma_client.embedding_model
    providers = batch_capable_providers()
    if model in providers:
        return model
    choices = " or ".join(f"--provider {p}" for p in sorted(providers))
    raise ValueError(
        f"Configured embedding model '{model}' has no Batch API support; "
        f"pass {choices} to select a manifest explicitly."
    )


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
    pending = [b for b in status.get("batches", []) if b.get("status") == "pending"]
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
            "Batch API indexing (OpenAI or Gemini, ~50% of realtime cost):\n"
            "  zotero-mcp update-db --batch            Submit embeddings asynchronously\n"
            "  zotero-mcp update-db --batch --auto-loop  Submit, then poll and import\n"
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
    update_db_parser.add_argument("--allow-mass-deletion", action="store_true",
                                 help="One-run opt-in when the deletion pass would remove "
                                      "a large share of the library's indexed documents "
                                      "(e.g. after intentionally purging the library)")
    update_db_parser.add_argument("--config-path",
                                 help="Path to semantic search configuration file")
    update_db_parser.add_argument("--db-path",
                                 help="Path to Zotero database file (zotero.sqlite), overrides config")
    update_db_parser.add_argument("--extraction-workers", type=int, metavar="N",
                                 help="Parse N attachments in parallel during --fulltext "
                                      "(default: semantic_search.extraction.workers, or 1 for "
                                      "sequential). Capped at the CPU count")
    update_db_parser.add_argument("--clear-fulltext-cache", action="store_true",
                                 help="Discard the transient extracted-fulltext cache before "
                                      "running, forcing every attachment to be re-parsed")
    batch_group = update_db_parser.add_mutually_exclusive_group()
    batch_group.add_argument("--batch", dest="use_batch", action="store_true",
                             help="Submit embeddings through the configured provider's asynchronous "
                                  "Batch API (pair with --batch-provider to choose explicitly)")
    batch_group.add_argument("--no-batch", dest="use_batch", action="store_false",
                             help="Use realtime embeddings even if a Batch API is enabled in config")
    update_db_parser.set_defaults(use_batch=None)
    update_db_parser.add_argument("--batch-provider", choices=BATCH_PROVIDERS, default=None,
                                 help="Batch provider to use with --batch (default: inferred from the "
                                      "configured embedding model)")
    update_db_parser.add_argument("--batch-max-tokens", type=int, default=None, metavar="N",
                                 help="Cap estimated tokens enqueued with the provider at once; "
                                      "chunks beyond it are held back and submitted as earlier ones finish")
    update_db_parser.add_argument("--batch-max-requests", type=int, default=None, metavar="N",
                                 help="Cap requests per uploaded batch file")
    update_db_parser.add_argument("--auto-loop", action="store_true",
                                 help="After submitting, keep polling, importing completed batches and "
                                      "submitting held-back chunks until the run finishes")
    update_db_parser.add_argument("--poll-interval", type=int, default=60, metavar="SECONDS",
                                 help="Seconds between --auto-loop polls (default: 60)")

    # Deprecated per-provider forms, kept working so existing scripts and
    # docs do not break. --batch/--batch-provider is the supported spelling.
    openai_batch_group = update_db_parser.add_mutually_exclusive_group()
    openai_batch_group.add_argument("--openai-batch", dest="openai_batch", action="store_true",
                                   help="Deprecated: use --batch --batch-provider openai")
    openai_batch_group.add_argument("--no-openai-batch", dest="openai_batch", action="store_false",
                                   help="Deprecated: use --no-batch")
    update_db_parser.set_defaults(openai_batch=None)
    gemini_batch_group = update_db_parser.add_mutually_exclusive_group()
    gemini_batch_group.add_argument("--gemini-batch", dest="gemini_batch", action="store_true",
                                   help="Deprecated: use --batch --batch-provider gemini")
    gemini_batch_group.add_argument("--no-gemini-batch", dest="gemini_batch", action="store_false",
                                   help="Deprecated: use --no-batch")
    update_db_parser.set_defaults(gemini_batch=None)

    # Batch lifecycle commands. The openai-batch-* spellings shipped in 0.10.0
    # stay as aliases pinned to --provider openai.
    for name, helptext in (
        ("batch-status", "Show embedding Batch API status"),
        ("openai-batch-status", "Deprecated alias for batch-status --provider openai"),
    ):
        sp = subparsers.add_parser(name, help=helptext)
        sp.add_argument("--batch-id", action="append",
                        help="Specific batch ID to inspect; can be repeated")
        sp.add_argument("--provider", choices=BATCH_PROVIDERS, default=None,
                        help="Which provider's manifests to read (default: inferred from config)")
        sp.add_argument("--config-path", help="Path to semantic search configuration file")

    for name, helptext in (
        ("batch-import", "Import completed batch embeddings"),
        ("openai-batch-import", "Deprecated alias for batch-import --provider openai"),
    ):
        sp = subparsers.add_parser(name, help=helptext)
        sp.add_argument("--batch-id", action="append",
                        help="Specific batch ID to import; can be repeated")
        sp.add_argument("--provider", choices=BATCH_PROVIDERS, default=None,
                        help="Which provider's manifests to read (default: inferred from config)")
        sp.add_argument("--config-path", help="Path to semantic search configuration file")

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

    # Schema refresh command
    subparsers.add_parser(
        "schema-refresh",
        help="Refresh the cached Zotero base-field schema from the server now")

    # Agent skill installation
    skill_parser = subparsers.add_parser(
        "install-skill",
        help="Install the zotero-cli agent skill into whatever agent harness "
             "is set up here (Claude Code, Cursor, Windsurf, AGENTS.md, Gemini)")
    skill_parser.add_argument(
        "--target", action="append", metavar="NAME",
        help="Install into this harness instead of auto-detecting. Repeatable. "
             "One of: claude, claude-user, cursor, windsurf, agents, gemini")
    skill_parser.add_argument(
        "--root", help="Directory to treat as the project root (default: cwd)")
    skill_parser.add_argument(
        "--list-targets", action="store_true",
        help="Show every supported harness and whether it is detected here")
    skill_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files. In a shared instructions file only the "
             "managed block changes. No backup is kept.")

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

    elif args.command == "install-skill":
        from pathlib import Path as _Path

        from zotero_mcp.skill_install import (
            TARGETS,
            detect_targets,
            format_results,
            install_skill,
        )

        root = _Path(args.root) if args.root else _Path.cwd()

        if args.list_targets:
            detected = set(detect_targets(root))
            print(f"Agent harnesses, checked against {root}:\n")
            for name, spec in TARGETS.items():
                mark = "detected" if name in detected else "-"
                print(f"  {name:<14} {mark:<10} {spec['label']}")
            print("\nRun `zotero-mcp install-skill` to install into every "
                  "detected one, or `--target NAME` to choose.")
            sys.exit(0)

        results = install_skill(
            targets=args.target, root=root, force=args.force,
        )
        message = format_results(results, root)
        failed = [r for r in results if r.status in ("error", "skipped")]
        print(message, file=sys.stderr if (failed and not any(r.ok for r in results))
              else sys.stdout)
        sys.exit(1 if (failed and not any(r.ok for r in results)) else 0)

    elif args.command == "schema-refresh":
        from zotero_mcp import schema
        before = schema.get_table().get("version")
        if schema.refresh(force=True) == "offline":
            print(
                "Could not reach the Zotero schema server; keeping the current "
                f"copy (version {before}).",
                file=sys.stderr,
            )
            sys.exit(1)
        after = schema.get_table().get("version")
        if after != before:
            print(f"Zotero schema refreshed: version {before} -> {after}.")
        else:
            print(f"Zotero schema already current (version {after}).")
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
                print(f"  Passage chunking: {_format_chunking_status(status)}")

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

        if getattr(args, "clear_fulltext_cache", False):
            from zotero_mcp import fulltext_cache

            removed = fulltext_cache.clear_all(config_path=str(config_path))
            print(f"Cleared transient fulltext cache ({removed} entries)")

        try:
            # Create semantic search instance with optional db_path override
            search = create_semantic_search(
                str(config_path),
                db_path=db_path,
                extraction_workers=getattr(args, "extraction_workers", None),
            )
            if args.use_batch is not None and (args.openai_batch is not None or args.gemini_batch is not None):
                print(
                    "Error: --batch/--no-batch cannot be combined with the deprecated "
                    "--openai-batch/--gemini-batch flags. Use --batch --batch-provider NAME.",
                    file=sys.stderr,
                )
                sys.exit(1)
            for flag, provider in (("--openai-batch", "openai"), ("--gemini-batch", "gemini")):
                if getattr(args, f"{provider}_batch") is True and (
                    search.chroma_client.embedding_model != provider
                ):
                    print(
                        f"Error: {flag} requires ZOTERO_EMBEDDING_MODEL={provider}",
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
                batch_max_tokens=args.batch_max_tokens,
                batch_max_requests=args.batch_max_requests,
                auto_loop=args.auto_loop,
                batch_poll_interval=args.poll_interval,
                allow_mass_deletion=args.allow_mass_deletion,
            )

            _print_update_stats(stats)

            if stats.get('error'):
                print(f"Error: {stats['error']}")
                sys.exit(1)

            # Best-effort sweep so the transient cache cannot outlive the runs
            # that fill it. Per-item eviction already covers everything that
            # embeds successfully; this catches what it structurally cannot —
            # entries abandoned by an interrupted run, entries whose
            # attachment has since been deleted, and .txt files orphaned by a
            # crash between writing the text and saving the index. Never fatal:
            # the update itself has already succeeded by this point.
            try:
                from zotero_mcp import fulltext_cache

                purged = fulltext_cache.purge_stale(config_path=str(config_path))
                if purged:
                    print(f"Purged {purged} stale fulltext cache entries")
            except Exception as e:
                print(f"Note: could not purge stale fulltext cache entries ({e})")

        except Exception as e:
            print(f"Error updating database: {e}")
            sys.exit(1)

    elif args.command in ("batch-status", "openai-batch-status"):
        setup_zotero_environment()

        from zotero_mcp.semantic_search import create_semantic_search

        config_path = _semantic_config_path(args.config_path)
        try:
            search = create_semantic_search(str(config_path))
            provider = "openai" if args.command == "openai-batch-status" else args.provider
            provider = provider or _detect_batch_provider(search)
            status = search._get_batch_status(provider, batch_ids=args.batch_id)
            _print_batch_status(status, provider)
        except Exception as e:
            print(f"Error getting batch status: {e}")
            sys.exit(1)

    elif args.command in ("batch-import", "openai-batch-import"):
        setup_zotero_environment()

        from zotero_mcp.semantic_search import create_semantic_search

        config_path = _semantic_config_path(args.config_path)
        try:
            search = create_semantic_search(str(config_path))
            provider = "openai" if args.command == "openai-batch-import" else args.provider
            provider = provider or _detect_batch_provider(search)
            stats = search._import_batch(provider, batch_ids=args.batch_id)
            _print_batch_import(stats, provider)
        except Exception as e:
            print(f"Error importing batch: {e}")
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
            # Create semantic search instance
            search = create_semantic_search(str(config_path))

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
            print(f"- Passage chunking: {_format_chunking_status(status)}")

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

        # Batch size for paginated collection scans (see _iter_all_metadatas).
        # Keeps each col.get() well under SQLite's bound-variable ceiling
        # regardless of collection size.
        DB_INSPECT_BATCH_SIZE = 500

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

            def _iter_all_metadatas(batch_size=DB_INSPECT_BATCH_SIZE, include_documents=False):
                """Paginate through the whole collection in bounded batches.

                A single unbounded ``col.get(include=[...])`` call (no limit/offset)
                asks Chroma's SQLite backend to bind one parameter per row for the
                entire collection; past roughly a few tens of thousands of rows this
                exceeds SQLite's bound-variable ceiling and raises
                ``too many SQL variables``. Fetching in small batches keeps every
                query well under that limit regardless of collection size, and lets
                filtering scan the *whole* collection instead of silently being
                limited to whatever the first raw batch happened to contain.
                """
                inc = ["metadatas", "documents"] if include_documents else ["metadatas"]
                total = col.count()
                offset = 0
                while offset < total:
                    batch = col.get(limit=batch_size, offset=offset, include=inc)
                    metas = batch.get("metadatas", [])
                    if not metas:
                        break
                    docs = batch.get("documents", [None] * len(metas)) if include_documents else [None] * len(metas)
                    for m, d in zip(metas, docs):
                        yield (m or {}), d
                    offset += batch_size

            if args.stats:
                # Show aggregate stats (merged from former db-stats).
                #
                # Single streaming pass over _iter_all_metadatas(): only the
                # small aggregates below (counters) are held in memory, never
                # a list of the collection's ~100k+ metadata dicts.
                print("=== Semantic DB Inspection (Stats) ===")
                info = client.get_collection_info()
                print(f"Collection: {info.get('name')} @ {info.get('persist_directory')}")
                print(f"Count: {info.get('count')}")

                ct_types = Counter()
                ct_titles = Counter()
                coverage = {}
                for m, _ in _iter_all_metadatas():
                    m = m or {}
                    t = m.get("item_type", "") or "(missing)"
                    ct_types[t] += 1

                    title = m.get("title", "")
                    if title:
                        ct_titles[title] += 1

                    cov = coverage.setdefault(t, {"total": 0, "with_fulltext": 0, "pdf": 0, "html": 0})
                    cov["total"] += 1
                    if m.get("has_fulltext"):
                        cov["with_fulltext"] += 1
                        src = (m.get("fulltext_source") or "").lower()
                        if src == "pdf":
                            cov["pdf"] += 1
                        elif src == "html":
                            cov["html"] += 1

                print("Item types:")
                for t, c in ct_types.most_common(20):
                    print(f"  {t or '(missing)'}: {c}")

                print("Fulltext coverage (by type):")
                for t, cov in coverage.items():
                    print(f"  {t}: {cov['with_fulltext']}/{cov['total']} (pdf:{cov['pdf']}, html:{cov['html']})")

                common = ct_titles.most_common(10)
                if common:
                    print("Common titles:")
                    for t, c in common:
                        print(f"  {t[:80]}{'...' if len(t)>80 else ''}: {c}")
                return

            print("=== Semantic DB Inspection ===")
            total = client.get_collection_info().get("count", 0)
            print(f"Total documents: {total}")
            print(f"Showing up to: {args.limit}")

            # Scan the whole collection in batches (not just the first raw batch),
            # so --filter actually finds matches wherever they live in a large
            # collection instead of only checking whatever `limit` records the
            # backend happened to return first.
            shown = 0
            for meta, doc in _iter_all_metadatas(include_documents=args.show_documents):
                title = meta.get("title", "")
                creators = meta.get("creators", "")
                if args.filter_text:
                    needle = args.filter_text.lower()
                    if needle not in (title or "").lower() and needle not in (creators or "").lower():
                        continue
                print(f"- {title} | {creators}")
                if args.show_documents:
                    full = (doc or "").strip()
                    snippet = full[:200].replace("\n", " ")
                    if snippet:
                        print(f"  doc: {snippet}{'...' if len(full) > 200 else ''}")
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
        # Re-apply the toolset profile now that the transport is known. The
        # import-time call in server.py assumed stdio; an HTTP transport also
        # needs the ChatGPT connector tools. setup_zotero_environment() runs
        # first so a ZOTERO_MCP_TOOLSETS set via the config file is honoured.
        from zotero_mcp.toolsets import UnknownToolsetError, apply_toolsets
        try:
            apply_toolsets(mcp, transport=transport)
        except UnknownToolsetError as e:
            print(f"❌ {e}")
            sys.exit(1)
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
