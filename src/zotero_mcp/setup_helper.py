#!/usr/bin/env python

"""
Setup helper for zotero-mcp.

This script provides utilities to automatically configure zotero-mcp
by finding the installed executable and updating Claude Desktop's config.
"""

import argparse
import getpass
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse


def _obfuscate_sensitive(value: str | None, keep_chars: int = 4) -> str:
    """Obfuscate sensitive values for terminal display."""
    if not value:
        return "Not provided"
    if len(value) <= keep_chars:
        return "*" * len(value)
    return value[:keep_chars] + "*" * (len(value) - keep_chars)


def _restrict_file_permissions(path) -> None:
    """Best-effort ``chmod 600`` on a config file that may hold credentials.

    Config files written here can contain ZOTERO_API_KEY and OpenAI/Gemini
    keys; default umask often leaves them world-readable, which matters on
    shared hosts / synced ~/.config. No-op on platforms without POSIX
    permissions (Windows).
    """
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def find_executable():
    """Find the full path to the zotero-mcp executable."""
    # Try to find the executable in the PATH
    exe_name = "zotero-mcp"
    if sys.platform == "win32":
        exe_name += ".exe"

    exe_path = shutil.which(exe_name)
    if exe_path:
        print(f"Found zotero-mcp in PATH at: {exe_path}")
        return exe_path

    # If not found in PATH, try to find it in common installation directories
    potential_paths = []

    # User site-packages
    import site
    for site_path in site.getsitepackages():
        potential_paths.append(Path(site_path) / "bin" / exe_name)

    # User's home directory
    potential_paths.append(Path.home() / ".local" / "bin" / exe_name)

    # Virtual environment
    if "VIRTUAL_ENV" in os.environ:
        potential_paths.append(Path(os.environ["VIRTUAL_ENV"]) / "bin" / exe_name)

    # Additional common locations
    if sys.platform == "darwin":  # macOS
        potential_paths.append(Path("/usr/local/bin") / exe_name)
        potential_paths.append(Path("/opt/homebrew/bin") / exe_name)

    for path in potential_paths:
        if path.exists() and os.access(path, os.X_OK):
            print(f"Found zotero-mcp at: {path}")
            return str(path)

    # If still not found, search in common directories
    print("Searching for zotero-mcp in common locations...")
    try:
        # On Unix-like systems, try using the 'find' command
        if sys.platform != 'win32':
            import subprocess
            result = subprocess.run(
                ["find", os.path.expanduser("~"), "-name", "zotero-mcp", "-type", "f", "-executable"],
                capture_output=True, text=True, timeout=10
            )
            paths = result.stdout.strip().split('\n')
            if paths and paths[0]:
                print(f"Found zotero-mcp at {paths[0]}")
                return paths[0]
    except Exception as e:
        print(f"Error searching for zotero-mcp: {e}")

    print("Warning: Could not find zotero-mcp executable.")
    print("Make sure zotero-mcp is installed and in your PATH.")
    return None


CLAUDE_CONFIG_FILENAME = "claude_desktop_config.json"

# Directory names used by the different Claude Desktop builds. "Claude" is the
# classic/documented one, "Claude Desktop" has been probed here for a long time,
# and "Claude-3p" is the alternate build reported in issue #392 (seen on Windows
# under %LOCALAPPDATA% and on macOS under ~/Library/Application Support).
CLAUDE_APP_DIR_NAMES = ("Claude", "Claude Desktop", "Claude-3p")


def claude_config_candidates() -> list:
    """Return every known Claude Desktop config location for this platform.

    The list is ordered by likelihood but is used for *probing*: we only ever
    write to a candidate that already exists (see ``find_all_claude_configs``),
    so listing an extra directory costs nothing. Issue #392: some builds keep
    their config in ``Claude-3p`` rather than ``Claude``, and writing to the
    wrong one silently has no effect.
    """
    config_paths = []

    # macOS
    if sys.platform == "darwin":
        app_support = Path.home() / "Library" / "Application Support"
        for name in CLAUDE_APP_DIR_NAMES:
            config_paths.append(app_support / name / CLAUDE_CONFIG_FILENAME)

    # Windows
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            for name in CLAUDE_APP_DIR_NAMES:
                config_paths.append(Path(appdata) / name / CLAUDE_CONFIG_FILENAME)
        # The Claude-3p build in issue #392 lives under %LOCALAPPDATA%, not
        # %APPDATA%, so probe the local app data root as well.
        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            config_paths.append(Path(localappdata) / "Claude-3p" / CLAUDE_CONFIG_FILENAME)
            # The Store build redirects %APPDATA%\\Claude into its package
            # folder (#26); a file written to the real %APPDATA% is never read.
            config_paths.extend(
                Path(localappdata).glob(
                    f"Packages/Claude_*/LocalCache/Roaming/Claude/{CLAUDE_CONFIG_FILENAME}"
                )
            )

    # Linux
    else:
        config_home = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        for name in CLAUDE_APP_DIR_NAMES:
            config_paths.append(Path(config_home) / name / CLAUDE_CONFIG_FILENAME)

    # De-duplicate while preserving order (e.g. APPDATA == LOCALAPPDATA).
    seen = set()
    unique_paths = []
    for path in config_paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def default_claude_config_path() -> Path:
    """Path used when no Claude Desktop config exists yet (fresh install)."""
    if sys.platform == "darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Claude Desktop" / CLAUDE_CONFIG_FILENAME
    if sys.platform == "win32":  # Windows
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Claude Desktop" / CLAUDE_CONFIG_FILENAME
    # Linux and others
    config_home = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
    return Path(config_home) / "Claude Desktop" / CLAUDE_CONFIG_FILENAME


def find_existing_claude_configs(verbose: bool = False) -> list:
    """Return every candidate Claude Desktop config that actually exists."""
    found = [path for path in claude_config_candidates() if path.exists()]
    if verbose:
        for path in found:
            print(f"Found Claude Desktop config at: {path}")
    return found


def find_all_claude_configs(verbose: bool = False) -> list:
    """Return the config path(s) setup should write to (never empty).

    Every existing config is returned: a machine can have more than one Claude
    Desktop build installed and picking just one is how issue #392 happened
    (setup reported success against a file the running app never reads).
    If none exists we fall back to the historical default path.
    """
    found = find_existing_claude_configs(verbose=verbose)
    if found:
        if verbose and len(found) > 1:
            print(f"Multiple Claude Desktop configs detected ({len(found)}).")
        return found

    default_path = default_claude_config_path()
    if verbose:
        print(f"Claude Desktop config not found. Using default path: {default_path}")
    return [default_path]


def find_claude_config(verbose: bool = False):
    """Find the Claude Desktop config file path to read from.

    Returns the first existing candidate, or the platform default if none
    exists. Callers that *write* config should use ``find_all_claude_configs``
    so they do not miss a second installed build.
    """
    return find_all_claude_configs(verbose=verbose)[0]


def _prompt_positive_int(label: str, default: int, minimum: int = 1) -> int:
    """Ask for an integer of at least ``minimum``; Enter keeps ``default``."""
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid number")
            continue
        if value >= minimum:
            return value
        print(f"Please enter a number of at least {minimum}")


def _prompt_text(label: str, current: str | None, blank_means: str) -> str | None:
    """Ask for a string; Enter keeps ``current``, ``-`` clears it.

    Returns the value to store, or None for "use the default".
    """
    if current:
        raw = input(f"{label} [{current}] (Enter keeps it, '-' resets to default): ").strip()
        if raw == "":
            return current
        return None if raw == "-" else raw
    raw = input(f"{label} (leave blank for {blank_means}): ").strip()
    return raw or None


def _prompt_secret(label: str, current: str | None, what: str = "key") -> str:
    """Ask for a secret without echoing; Enter keeps ``current`` if there is one."""
    if current:
        raw = getpass.getpass(f"{label} (hidden, Enter keeps the stored {what}): ").strip()
        return raw or current
    return getpass.getpass(f"{label} (hidden): ").strip()


def _prompt_choice(label: str, valid: list[str], default: str | None) -> str:
    """Ask for one of ``valid``; Enter picks ``default`` when there is one."""
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if raw == "" and default:
            return default
        if raw in valid:
            return raw
        print("Please enter " + ", ".join(valid[:-1]) + f", or {valid[-1]}")


def _prompt_chunking(existing_semantic_config: dict | None, batch_enabled: bool) -> dict:
    """Ask whether to index documents as overlapping passages.

    Starts from the existing ``chunking`` block (hand-edited keys survive) and
    only asks for the sizes when chunking is switched on. Enabling or disabling
    it changes the record layout, so the user is told a rebuild is needed.
    """
    existing = dict((existing_semantic_config or {}).get("chunking") or {})
    was_enabled = bool(existing.get("enabled", False))
    chunking = {
        "enabled": was_enabled,
        "chunk_size": 1500,
        "overlap": 200,
        "max_chunks_per_item": 20,
    }
    chunking.update(existing)

    print("\n=== Passage Chunking ===")
    print("Without chunking, each item is one vector: the metadata plus the start of")
    print("the full text, cut at the embedding model's input limit. Text beyond that")
    print("is not searchable.")
    print("With chunking, each item is split into overlapping passages, one vector")
    print("each, so long PDFs stay searchable and results can quote the matching passage.")
    print("Warning: on a large library chunking multiplies the index: many more embeddings")
    print("to pay for and wait on, and the index sits in memory (about 6 GB per million")
    print("passages). Changing it needs 'zotero-mcp update-db --force-rebuild'.")
    print(
        "The Batch API is enabled here and ignores chunking; use 'update-db --no-batch'."
        if batch_enabled else "The Batch API ignores chunking."
    )
    hint = "Y/n" if was_enabled else "y/N"
    raw = input(f"Index full text as passages? [{hint}]: ").strip().lower()
    enabled = was_enabled if raw == "" else raw in ("y", "yes")
    chunking["enabled"] = enabled

    if enabled:
        chunking["chunk_size"] = _prompt_positive_int(
            "Passage size in characters", int(chunking["chunk_size"])
        )
        while True:
            chunking["overlap"] = _prompt_positive_int(
                "Overlap between passages in characters", int(chunking["overlap"]), minimum=0
            )
            if chunking["overlap"] < chunking["chunk_size"]:
                break
            print("Overlap must be smaller than the passage size")
        print("Max passages per item bounds how much of each document is searchable")
        print("(passage size x max passages). More passages cost more to embed.")
        chunking["max_chunks_per_item"] = _prompt_positive_int(
            "Max passages per item", int(chunking["max_chunks_per_item"])
        )
        print(
            f"That is up to {int(chunking['max_chunks_per_item']) * 10_000:,} embeddings "
            "per 10,000 items."
        )
    if enabled != was_enabled:
        print("Chunking changed: run 'zotero-mcp update-db --force-rebuild' to apply it.")
    return chunking


def setup_semantic_search(
    existing_semantic_config: dict | None = None,
    semantic_config_only_arg: bool = False,
    existing_db_path: str | None = None,
) -> tuple[dict, str | None]:
    """Interactive setup for semantic search configuration."""
    print("\n=== Semantic Search Configuration ===")

    if existing_semantic_config:
        # Same rows as `db-status`; never prints the API key.
        from zotero_mcp.semantic_summary import print_rows, semantic_config_rows

        print("Found existing semantic search configuration:")
        print_rows(
            semantic_config_rows(existing_semantic_config, existing_db_path),
            "  - ",
        )
        print("You can keep it or change it.")
        print("If you change to a new configuration, a database rebuild is advised.")
        print("Would you like to keep your existing configuration? (y/n): ", end="")
        if input().strip().lower() in ['y', 'yes']:
            return existing_semantic_config, existing_db_path

    print("Configure embedding models for semantic search over your Zotero library.")

    # Choose embedding model
    print("\nAvailable embedding models:")
    print("1. Default (all-MiniLM-L6-v2) - Free, runs locally")
    print("2. OpenAI - Better quality, requires API key")
    print("3. Gemini - Better quality, requires API key")
    print("4. Ollama - Local embeddings via Ollama API")

    prev = existing_semantic_config or {}
    prev_model = prev.get("embedding_model")
    prev_embedding = prev.get("embedding_config") or {}
    default_choice = {"default": "1", "openai": "2", "gemini": "3", "ollama": "4"}.get(prev_model)
    print()
    choice = _prompt_choice("Choose embedding model (1-4)", ["1", "2", "3", "4"], default_choice)
    # Stored values carry over only when the provider is unchanged: an OpenAI
    # key must never be offered to Gemini.
    def same(provider: str) -> bool:
        return prev_model == provider

    config = {}

    if choice == "1":
        config["embedding_model"] = "default"
        print("Using default embedding model (all-MiniLM-L6-v2)")

    elif choice == "2":
        config["embedding_model"] = "openai"

        # Choose OpenAI model
        print("\nOpenAI embedding models:")
        print("1. text-embedding-3-small (recommended, faster)")
        print("2. text-embedding-3-large (higher quality, slower)")

        prev_name = prev_embedding.get("model_name") if same("openai") else None
        default_model_choice = {"text-embedding-3-small": "1", "text-embedding-3-large": "2"}.get(prev_name)
        model_choice = _prompt_choice("Choose OpenAI model (1-2)", ["1", "2"], default_model_choice)

        if model_choice == "1":
            config["embedding_config"] = {"model_name": "text-embedding-3-small"}
        else:
            config["embedding_config"] = {"model_name": "text-embedding-3-large"}

        # Get API key
        api_key = _prompt_secret(
            "Enter your OpenAI API key", prev_embedding.get("api_key") if same("openai") else None
        )
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            print("Warning: No API key provided. Set OPENAI_API_KEY environment variable.")

        # Get optional base URL
        base_url = _prompt_text(
            "Custom OpenAI base URL", prev_embedding.get("base_url") if same("openai") else None, "default"
        )
        if base_url:
            config["embedding_config"]["base_url"] = base_url
            print(f"Using custom OpenAI base URL: {base_url}")
        else:
            print("Using default OpenAI base URL")

        existing_openai_batch = existing_semantic_config.get("openai_batch", {}) if existing_semantic_config else {}
        if existing_semantic_config and existing_semantic_config.get("embedding_model") == "openai":
            default_batch_enabled = bool(existing_openai_batch.get("enabled", False))
        else:
            default_batch_enabled = True
        default_hint = "Y/n" if default_batch_enabled else "y/N"
        print("\nOpenAI indexing mode:")
        print("Batch API lowers costs for database updates, but results are imported later after the batch completes.")
        print("Realtime API indexes immediately, but uses standard synchronous embedding pricing.")
        raw = input(f"Use OpenAI Batch API for database updates? [{default_hint}]: ").strip().lower()
        if raw == "":
            batch_enabled = default_batch_enabled
        else:
            batch_enabled = raw in ["y", "yes"]
        config["openai_batch"] = {"enabled": batch_enabled}
        if batch_enabled:
            print("OpenAI Batch API will be used by default for update-db.")
        else:
            print("Realtime OpenAI embeddings will be used by default for update-db.")

    elif choice == "3":
        config["embedding_model"] = "gemini"

        config["embedding_config"] = {"model_name": "gemini-embedding-001"}

        # Get API key
        api_key = _prompt_secret(
            "Enter your Gemini API key", prev_embedding.get("api_key") if same("gemini") else None
        )
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            print("Warning: No API key provided. Set GEMINI_API_KEY environment variable.")

        # Get optional base URL
        base_url = _prompt_text(
            "Custom Gemini base URL", prev_embedding.get("base_url") if same("gemini") else None, "default"
        )
        if base_url:
            config["embedding_config"]["base_url"] = base_url
            print(f"Using custom Gemini base URL: {base_url}")
        else:
            print("Using default Gemini base URL")

        existing_gemini_batch = existing_semantic_config.get("gemini_batch", {}) if existing_semantic_config else {}
        default_gemini_batch = bool(existing_gemini_batch.get("enabled", False))
        default_hint = "Y/n" if default_gemini_batch else "y/N"
        print("\nGemini indexing mode:")
        print("Batch API lowers costs for database updates, but results are imported later after the batch completes.")
        print("Note: Google marks the Gemini embeddings Batch API as experimental.")
        raw = input(f"Use Gemini Batch API for database updates? [{default_hint}]: ").strip().lower()
        if raw == "":
            gemini_batch_enabled = default_gemini_batch
        else:
            gemini_batch_enabled = raw in ["y", "yes"]
        config["gemini_batch"] = {"enabled": gemini_batch_enabled}
        if gemini_batch_enabled:
            print("Gemini Batch API will be used by default for update-db.")
        else:
            print("Realtime Gemini embeddings will be used by default for update-db.")

    elif choice == "4":
        config["embedding_model"] = "ollama"
        current_name = (prev_embedding.get("model_name") if same("ollama") else None) or "qwen3-embedding"
        model_name = input(f"Ollama embedding model name [{current_name}]: ").strip()
        config["embedding_config"] = {"model_name": model_name or current_name}

        base_url = _prompt_text(
            "Ollama base URL", prev_embedding.get("base_url") if same("ollama") else None,
            "http://localhost:11434",
        )
        if base_url:
            config["embedding_config"]["base_url"] = base_url
            print(f"Using custom Ollama base URL: {base_url}")
        else:
            print("Using default Ollama base URL: http://localhost:11434")

    # Configure update frequency
    print("\n=== Database Update Configuration ===")
    print("Configure how often the semantic search database is updated:")
    print("1. Manual - Update only when you run 'zotero-mcp update-db'")
    print("2. Auto - Automatically update on server startup")
    print("3. Daily - Automatically update once per day")
    print("4. Every N days - Automatically update every N days")

    prev_update = prev.get("update_config") or {}
    prev_freq = str(prev_update.get("update_frequency") or "")
    default_update = (
        {"manual": "1", "startup": "2", "daily": "3"}.get(prev_freq)
        or ("4" if prev_freq.startswith("every_") else None)
    )
    print()
    update_choice = _prompt_choice("Choose update frequency (1-4)", ["1", "2", "3", "4"], default_update)

    update_config = {}

    if update_choice == "1":
        update_config = {
            "auto_update": False,
            "update_frequency": "manual"
        }
        print("Database will only be updated manually.")
    elif update_choice == "2":
        update_config = {
            "auto_update": True,
            "update_frequency": "startup"
        }
        print("Database will be updated every time the server starts.")
    elif update_choice == "3":
        update_config = {
            "auto_update": True,
            "update_frequency": "daily",
            "update_days": None
        }
        print("Database will be updated once per day.")
    elif update_choice == "4":
        prev_days = prev_update.get("update_days")
        days = _prompt_positive_int(
            "Number of days between updates",
            prev_days if isinstance(prev_days, int) and prev_days > 0 else 7,
        )

        update_config = {
            "auto_update": True,
            "update_frequency": f"every_{days}",
            "update_days": days
        }
        print(f"Database will be updated every {days} days.")

    # Configure extraction settings
    print("\n=== Content Extraction Settings ===")
    print("Set a page cap for PDF extraction to balance speed vs. coverage.")
    print("Press Enter to use the default.")
    default_pdf_max = existing_semantic_config.get("extraction", {}).get("pdf_max_pages", 10) if existing_semantic_config else 10
    while True:
        raw = input(f"PDF max pages [{default_pdf_max}]: ").strip()
        if raw == "":
            pdf_max_pages = default_pdf_max
            break
        try:
            pdf_max_pages = int(raw)
            if pdf_max_pages > 0:
                break
            print("Please enter a positive integer")
        except ValueError:
            print("Please enter a valid number")

    print("\nAttachment parsing can run across several CPU cores.")
    print("1 keeps the classic sequential behaviour; higher values speed up")
    print("the extraction phase of 'update-db --fulltext' on large libraries.")
    default_workers = (
        existing_semantic_config.get("extraction", {}).get("workers", 1)
        if existing_semantic_config
        else 1
    )
    while True:
        raw = input(f"Parallel extraction workers [{default_workers}]: ").strip()
        if raw == "":
            extraction_workers = default_workers
            break
        try:
            extraction_workers = int(raw)
            if extraction_workers > 0:
                break
            print("Please enter a positive integer")
        except ValueError:
            print("Please enter a valid number")

    batch_enabled = bool(
        config.get("openai_batch", {}).get("enabled")
        or config.get("gemini_batch", {}).get("enabled")
    )
    chunking_config = _prompt_chunking(existing_semantic_config, batch_enabled)

    # Configure Zotero database path
    print("\n=== Zotero Database Path ===")
    print("By default, zotero-mcp auto-detects the Zotero database location.")
    print("If Zotero is installed in a custom location, you can specify the path here.")
    default_db_path = existing_db_path or (existing_semantic_config.get("zotero_db_path", "") if existing_semantic_config else "")
    db_path_hint = default_db_path if default_db_path else "auto-detect"
    raw_db_path = input(f"Zotero database path [{db_path_hint}]: ").strip()

    # Validate path if provided
    zotero_db_path = None
    if raw_db_path:
        db_file = Path(raw_db_path)
        if db_file.exists() and db_file.is_file():
            zotero_db_path = str(db_file)
            print(f"Using custom Zotero database: {zotero_db_path}")
        else:
            print(f"Warning: File not found at '{raw_db_path}'. Using auto-detect instead.")
    elif default_db_path:
        # Keep existing custom path if user just pressed Enter
        zotero_db_path = default_db_path
        print(f"Keeping existing database path: {zotero_db_path}")
    else:
        print("Using auto-detect for Zotero database location.")

    config["update_config"] = update_config
    # Merge rather than replace: setup only prompts for the page cap, and
    # rewriting the whole section would silently drop hand-edited keys such
    # as attachment_priority and fulltext_display_max_pages.
    extraction_config = dict(
        (existing_semantic_config or {}).get("extraction") or {}
    )
    extraction_config["pdf_max_pages"] = pdf_max_pages
    extraction_config["workers"] = extraction_workers
    config["extraction"] = extraction_config
    # Web-API users: index fulltext from Zotero's server-side extraction by
    # default. Users can set this to false to keep the old metadata-only
    # behavior. The flag is ignored in local mode (ZOTERO_LOCAL=true uses
    # local sqlite extraction).
    config.setdefault("include_fulltext", True)
    # Discoverable defaults for the two retrieval-quality knobs (both off so
    # the base experience is unchanged). Preserve any existing user values.
    #   reranker: local cross-encoder re-rank of candidates for higher
    #             precision (needs sentence-transformers; adds a model load).
    #   chunking: index each item as overlapping passages so search returns
    #             grounded quotes and long PDFs stay searchable. Enabling it
    #             requires a one-time `update-db --force-rebuild`. It has no
    #             effect while `openai_batch.enabled` is true: that path
    #             indexes item-level and update-db warns about it (#416).
    if existing_semantic_config and existing_semantic_config.get("reranker"):
        config["reranker"] = existing_semantic_config["reranker"]
    else:
        config.setdefault("reranker", {
            "enabled": False,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "candidate_multiplier": 3,
        })
    # Setup has no prompt for it; dropping it would point the index back at
    # the default path (#617).
    if existing_semantic_config and existing_semantic_config.get("persist_directory"):
        config["persist_directory"] = existing_semantic_config["persist_directory"]
    config["chunking"] = chunking_config
    return config, zotero_db_path


def save_semantic_search_config(
    config: dict,
    semantic_config_path: Path,
    zotero_db_path: str | None = None,
) -> bool:
    """Save semantic search configuration to file."""
    try:
        # Ensure config directory exists
        semantic_config_dir = semantic_config_path.parent
        from zotero_mcp.utils import ensure_private_dir
        ensure_private_dir(semantic_config_dir)

        # Load existing config or create new one
        full_semantic_config = {}
        if semantic_config_path.exists():
            try:
                with open(semantic_config_path) as f:
                    full_semantic_config = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Existing semantic search config file is invalid JSON, creating new one")

        # Add semantic search config
        full_semantic_config["semantic_search"] = config

        # Store zotero_db_path at the top level (not under semantic_search)
        if zotero_db_path:
            full_semantic_config["zotero_db_path"] = zotero_db_path

        # Write config
        with open(semantic_config_path, 'w') as f:
            json.dump(full_semantic_config, f, indent=2)
        _restrict_file_permissions(semantic_config_path)

        print(f"Semantic search configuration saved to: {semantic_config_path}")
        return True

    except Exception as e:
        print(f"Error saving semantic search config: {e}")
        return False

def load_top_level_db_path(config_path: Path) -> str | None:
    """Load the top-level ``zotero_db_path`` from the config file."""
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            return json.load(f).get("zotero_db_path")
    except Exception:
        return None


def load_semantic_search_config(semantic_config_path: Path) -> dict:
    """Load existing semantic search configuration."""
    if not semantic_config_path.exists():
        return {}

    try:
        with open(semantic_config_path) as f:
            full_semantic_config = json.load(f)
        return full_semantic_config.get("semantic_search", {})
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse config file as JSON: {e}")
        return {}
    except Exception as e:
        print(f"Warning: Could not read config file: {e}")
        return {}


WEBDAV_KEYS = ("ZOTERO_WEBDAV_URL", "ZOTERO_WEBDAV_USERNAME", "ZOTERO_WEBDAV_PASSWORD")


def _webdav_values(env) -> dict | None:
    """The three WebDAV settings in *env*, or None unless all are set (as webdav.py reads them)."""
    from zotero_mcp.webdav import _PLACEHOLDER_PREFIX

    values = {key: str(env.get(key) or "").strip() for key in WEBDAV_KEYS}
    if all(v and not v.startswith(_PLACEHOLDER_PREFIX) for v in values.values()):
        return values
    return None


def _existing_webdav(config_paths) -> dict | None:
    """The WebDAV settings in the first of *config_paths* that has all three."""
    for path in config_paths:
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            env = cfg.get("client_env") or cfg.get("mcpServers", {}).get("zotero", {}).get("env") or {}
            found = _webdav_values(env)
        except Exception:
            continue
        if found:
            return found
    return None


def _webdav_env(existing_env, webdav: dict | None) -> dict:
    """The WebDAV settings to write into a config's env section.

    *webdav* is setup's answer and replaces what the config had; ``{}`` means
    no WebDAV. With None, setup did not ask, and the settings already in
    *existing_env* are kept as they are.
    """
    if webdav is not None:
        return webdav
    if not isinstance(existing_env, dict):
        return {}
    return {key: existing_env[key] for key in WEBDAV_KEYS if key in existing_env}


def _zotero_folder_url(address: str) -> str:
    """The folder Zotero keeps its files in, from the address in Zotero's sync settings.

    Zotero's settings drop a trailing /zotero from what is typed and add zotero/ to the
    address; the upload writes <KEY>.zip straight into the folder it is given, so it
    needs that folder.
    """
    url = re.sub(r"/zotero/?$|/$", "", address.strip())
    if "://" not in url:  # Zotero's settings pick the scheme in a separate menu
        url = "https://" + url
    return url + "/zotero/"


def _webdav_address_problem(url: str) -> str | None:
    """Why *url* cannot be the WebDAV folder, or None when it can."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return "it is not an http or https address"
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "it is not an http or https address"
    if parsed.username or parsed.password:
        return "the username and the password are asked for next, so leave them out of it"
    return None


def setup_webdav(existing: dict | None) -> dict | None:
    """Ask whether attachment files sync over WebDAV.

    Returns the three ZOTERO_WEBDAV_* settings to write, ``{}`` for no WebDAV,
    or None to leave each config as it is: at the end of input, when nothing
    is stored and the question is passed with Enter, or when an answer is
    left blank. Complete settings in the environment are used without asking,
    as ZOTERO_API_KEY is. *existing* holds the stored settings; they are the
    defaults, so Enter keeps them.
    """
    from_env = _webdav_values(os.environ)
    if from_env:
        print("\nUsing the WebDAV settings in the environment (ZOTERO_WEBDAV_*).")
        return from_env

    print("\n=== WebDAV File Sync ===")
    print("If Zotero syncs your attachment files over WebDAV (Zotero settings > Sync >")
    print("File Syncing), files added through the web API can go straight to that server.")
    print("Otherwise they go to Zotero Storage and count against its quota.")
    if existing and _webdav_address_problem(existing["ZOTERO_WEBDAV_URL"]):
        existing = None  # not shown: it may hold a login. Enter still leaves it in the config
    stored = existing or {}
    stored_url = stored.get("ZOTERO_WEBDAV_URL")
    stored_username = stored.get("ZOTERO_WEBDAV_USERNAME")
    hint = "Y/n" if existing else "y/N"
    try:
        while True:
            raw = input(f"Do you sync attachment files over WebDAV? [{hint}]: ").strip().lower()
            if raw in ("", "y", "yes", "n", "no"):
                break
            print("Please enter y or n")
        if raw in ("n", "no"):
            return {}
        if raw == "" and not existing:
            return None
        while True:
            if existing:
                address = input(f"WebDAV address, as in Zotero's settings [{stored_url}]: ").strip()
            else:
                address = input(
                    "WebDAV address as in Zotero's settings, with https:// or http:// "
                    "(e.g. https://dav.example.com): "
                ).strip()
            if not address:
                url = stored_url  # Enter keeps the stored folder as it is
                break
            url = _zotero_folder_url(address)
            problem = _webdav_address_problem(url)
            if problem is None:
                break
            # The address is not repeated: it may hold a password.
            print(f"That address cannot be used: {problem}.")
        if not url:
            print("WebDAV settings not changed: no address was given.")
            return None
        if existing:
            username = input(f"WebDAV username [{stored_username}]: ").strip() or stored_username
        else:
            username = input("WebDAV username: ").strip()
        password = _prompt_secret("WebDAV password", stored.get("ZOTERO_WEBDAV_PASSWORD"), "password")
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not (username and password):
        print("WebDAV settings not changed: the username and the password are both needed.")
        return None
    print(f"Files will go to: {url}")
    return {
        "ZOTERO_WEBDAV_URL": url,
        "ZOTERO_WEBDAV_USERNAME": username,
        "ZOTERO_WEBDAV_PASSWORD": password,
    }


def update_claude_config(config_path, zotero_mcp_path, local=True, api_key=None, library_id=None, library_type="user", semantic_config=None, webdav=None):
    """Update Claude Desktop config to add zotero-mcp."""
    # Create directory if it doesn't exist
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new one. Claude Desktop writes this file
    # as UTF-8; without an explicit encoding Windows reads it with the locale
    # code page and the rewrite below would garble other servers' entries.
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            print(f"Loaded existing config from: {config_path}")
        except json.JSONDecodeError:
            print(f"Error: Config file at {config_path} is not valid JSON. Creating new config.")
            config = {}
    else:
        print(f"Creating new config file at: {config_path}")
        config = {}

    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Create environment settings based on local vs web API
    env_settings = {
        "ZOTERO_LOCAL": "true" if local else "false"
    }
    # Setup changes the Zotero connection, not HTTP authentication. Preserve
    # even an invalid value so the serve-time validator can reject it safely.
    existing_zotero = config["mcpServers"].get("zotero", {})
    existing_env = existing_zotero.get("env", {}) if isinstance(existing_zotero, dict) else {}
    if isinstance(existing_env, dict) and "ZOTERO_MCP_AUTH_TOKEN" in existing_env:
        env_settings["ZOTERO_MCP_AUTH_TOKEN"] = existing_env["ZOTERO_MCP_AUTH_TOKEN"]

    # Add API key and library settings for web API
    if not local:
        if api_key:
            env_settings["ZOTERO_API_KEY"] = api_key
        if library_id:
            env_settings["ZOTERO_LIBRARY_ID"] = library_id
        if library_type:
            env_settings["ZOTERO_LIBRARY_TYPE"] = library_type
    # Without these a rebuilt env section sends uploads back to Zotero Storage.
    env_settings.update(_webdav_env(existing_env, webdav))

    # Add semantic search settings if provided
    if semantic_config:
        env_settings["ZOTERO_EMBEDDING_MODEL"] = semantic_config.get("embedding_model", "default")

        embedding_config = semantic_config.get("embedding_config", {})
        if semantic_config.get("embedding_model") == "openai":
            if api_key := embedding_config.get("api_key"):
                env_settings["OPENAI_API_KEY"] = api_key
            if model := embedding_config.get("model_name"):
                env_settings["OPENAI_EMBEDDING_MODEL"] = model
            if base_url := embedding_config.get("base_url"):
                env_settings["OPENAI_BASE_URL"] = base_url

        elif semantic_config.get("embedding_model") == "gemini":
            if api_key := embedding_config.get("api_key"):
                env_settings["GEMINI_API_KEY"] = api_key
            if model := embedding_config.get("model_name"):
                env_settings["GEMINI_EMBEDDING_MODEL"] = model
            if base_url := embedding_config.get("base_url"):
                env_settings["GEMINI_BASE_URL"] = base_url

        elif semantic_config.get("embedding_model") == "ollama":
            if model := embedding_config.get("model_name"):
                env_settings["OLLAMA_EMBEDDING_MODEL"] = model
            if base_url := embedding_config.get("base_url"):
                env_settings["OLLAMA_BASE_URL"] = base_url

    # Add or update zotero config
    config["mcpServers"]["zotero"] = {
        "command": zotero_mcp_path,
        "env": env_settings
    }

    # Write updated config
    try:
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        _restrict_file_permissions(config_path)
        print(f"\nSuccessfully wrote config to: {config_path}")
    except Exception as e:
        print(f"Error writing config file: {str(e)}")
        return False

    return config_path


def _write_standalone_config(local: bool, api_key: str, library_id: str, library_type: str, semantic_config: dict, no_claude: bool = False, webdav: dict | None = None) -> Path:
    """Write a central config file used by semantic search and provide client env."""
    cfg_dir = Path.home() / ".config" / "zotero-mcp"
    from zotero_mcp.utils import ensure_private_dir
    ensure_private_dir(cfg_dir)
    cfg_path = cfg_dir / "config.json"

    # Load or initialize
    full = {}
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                full = json.load(f)
        except Exception:
            full = {}

    # Store semantic config if provided
    if semantic_config:
        full["semantic_search"] = semantic_config

    # Provide a helper env section for web-based clients
    client_env = {
        "ZOTERO_LOCAL": "true" if local else "false"
    }
    existing_env = full.get("client_env", {})
    if isinstance(existing_env, dict) and "ZOTERO_MCP_AUTH_TOKEN" in existing_env:
        client_env["ZOTERO_MCP_AUTH_TOKEN"] = existing_env["ZOTERO_MCP_AUTH_TOKEN"]
    # Persist global guard to disable Claude detection/output if requested
    if no_claude:
        client_env["ZOTERO_NO_CLAUDE"] = "true"
    if not local:
        if api_key:
            client_env["ZOTERO_API_KEY"] = api_key
        if library_id:
            client_env["ZOTERO_LIBRARY_ID"] = library_id
        if library_type:
            client_env["ZOTERO_LIBRARY_TYPE"] = library_type
    client_env.update(_webdav_env(existing_env, webdav))

    full["client_env"] = client_env

    with open(cfg_path, 'w') as f:
        json.dump(full, f, indent=2)
    _restrict_file_permissions(cfg_path)

    return cfg_path


def main(cli_args=None):
    """Main function to run the setup helper."""
    parser = argparse.ArgumentParser(description="Configure zotero-mcp for Claude Desktop")
    parser.add_argument("--no-local", action="store_true", help="Configure for Zotero Web API instead of local API")
    parser.add_argument("--no-claude", action="store_true", help="Don't setup Claude Desktop config: instead store settings in config file.")
    parser.add_argument("--api-key",
                        help="Zotero API key (only needed with --no-local). Insecure on "
                             "multi-user systems — leaks via 'ps'/shell history; omit to be "
                             "prompted securely or set the ZOTERO_API_KEY env var.")
    parser.add_argument("--library-id", help="Zotero library ID (only needed with --no-local)")
    parser.add_argument("--library-type", choices=["user", "group"], default="user",
                        help="Zotero library type (only needed with --no-local)")
    parser.add_argument("--config-path", help="Path to Claude Desktop config file")
    parser.add_argument("--skip-semantic-search", action="store_true",
                        help="Skip semantic search configuration")
    parser.add_argument("--semantic-config-only", action="store_true",
                        help="Only configure semantic search, skip Zotero setup")
    parser.add_argument("--show-secrets", action="store_true",
                        help="Print secrets (e.g. the Zotero API key) in plaintext in the "
                             "setup summary. Off by default; values are masked.")

    # If this is being called from CLI with existing args
    if cli_args is not None and hasattr(cli_args, 'no_local'):
        args = cli_args
        print("Using arguments passed from command line")
    else:
        # Otherwise parse from command line
        args = parser.parse_args()
        print("Parsed arguments from command line")

    # Determine config path for semantic search
    semantic_config_dir = Path.home() / ".config" / "zotero-mcp"
    semantic_config_path = semantic_config_dir / "config.json"
    existing_semantic_config = load_semantic_search_config(semantic_config_path)
    existing_db_path = load_top_level_db_path(semantic_config_path)
    semantic_config_changed = False

    # Handle semantic search only configuration
    if args.semantic_config_only:
        print("Configuring semantic search only...")
        new_semantic_config, zotero_db_path = setup_semantic_search(existing_semantic_config, existing_db_path=existing_db_path)
        semantic_config_changed = existing_semantic_config != new_semantic_config
        # only save if semantic config changed
        if semantic_config_changed:
            if save_semantic_search_config(new_semantic_config, semantic_config_path, zotero_db_path):
                print("\nSemantic search configuration complete!")
                print(f"Configuration saved to: {semantic_config_path}")
                print("\nTo initialize the database, run: zotero-mcp update-db")
                return 0
            else:
                print("\nSemantic search configuration failed.")
                return 1
        else:
            print("\nSemantic search configuration left unchanged.")
            return 0

    # Find zotero-mcp executable
    exe_path = find_executable()
    if not exe_path:
        print("Error: Could not find zotero-mcp executable.")
        return 1
    print(f"Using zotero-mcp at: {exe_path}")

    # Find Claude Desktop config(s) unless --no-claude
    config_paths = []
    if not args.no_claude:
        if args.config_path:
            print(f"Using specified config path: {args.config_path}")
            config_paths = [Path(args.config_path)]
        else:
            config_paths = find_all_claude_configs(verbose=True)
        if not config_paths:
            print("Error: Could not determine Claude Desktop config path.")
            return 1

    # Update config
    use_local = not args.no_local
    api_key = args.api_key
    library_id = args.library_id
    library_type = args.library_type

    # Web mode needs an API key. Prefer the env var, then prompt securely —
    # avoid forcing it onto the command line (where it leaks via ps/history).
    if not use_local and not api_key:
        api_key = os.environ.get("ZOTERO_API_KEY")
        if not api_key:
            try:
                api_key = getpass.getpass(
                    "Enter your Zotero API key (hidden, input not echoed): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                api_key = None

    # Configure semantic search if not skipped
    if not args.skip_semantic_search:
        # if there is already a semantic search configuration in the config file:
        if existing_semantic_config:
            print("\nFound an exisiting semantic search configuration in the config file.")
            print("Would you like to reconfigure semantic search? (y/n): ", end="")
        # if otherwise, slightly different message...
        else:
            print("\nWould you like to configure semantic search? (y/n): ", end="")
        # Either way:
        if input().strip().lower() in ['y', 'yes']:
            new_semantic_config, zotero_db_path = setup_semantic_search(existing_semantic_config, existing_db_path=existing_db_path)
            if existing_semantic_config != new_semantic_config:
                semantic_config_changed = True
                existing_semantic_config = new_semantic_config  # Update the config to use
                save_semantic_search_config(existing_semantic_config, semantic_config_path, zotero_db_path)

    # WebDAV file sync, asked in web mode for a personal library (a group's
    # files always go to Zotero Storage). Otherwise each config keeps the
    # WebDAV settings it has. It is the last question: a script that pipes
    # answers to the ones above reaches end of input here, which changes
    # nothing.
    webdav = None
    if not use_local and library_type == "user":
        rewritten = [semantic_config_path] if args.no_claude else config_paths
        webdav = setup_webdav(_existing_webdav(rewritten))

    print("\nSetup with the following settings:")
    print(f"  Local API: {use_local}")
    if not use_local:
        print(f"  API Key: {_obfuscate_sensitive(api_key)}")
        print(f"  Library ID: {library_id or 'Not provided'}")
        print(f"  Library Type: {library_type}")
        if webdav is not None:
            print(f"  WebDAV file sync: {'yes' if webdav else 'no'}")

    # Use the potentially updated semantic config
    semantic_config = existing_semantic_config

    # Update configuration based on mode
    try:
        if args.no_claude:
            cfg_path = _write_standalone_config(
                local=use_local,
                api_key=api_key,
                library_id=library_id,
                library_type=library_type,
                semantic_config=semantic_config,
                no_claude=args.no_claude,
                webdav=webdav,
            )
            print("\nSetup complete (standalone/web mode)!")
            print(f"Config saved to: {cfg_path}")
            # Emit one-line client_env for easy copy/paste. Mask credentials
            # by default — single-line JSON is exactly what gets pasted into
            # bug reports / shared terminals; only print it in full on request.
            try:
                with open(cfg_path) as f:
                    full = json.load(f)
                client_env = full.get("client_env", {})
                show_secrets = getattr(args, "show_secrets", False)
                if show_secrets:
                    display_env = client_env
                else:
                    from zotero_mcp.cli import obfuscate_config_for_display

                    display_env = obfuscate_config_for_display(client_env)
                print("Client environment (single-line JSON):")
                print(json.dumps(display_env, separators=(',', ':')))
                if not show_secrets and display_env != client_env:
                    print("  (Secrets masked — re-run with --show-secrets to print them in full)")
            except Exception:
                pass
            if semantic_config_changed:
                print("\nNote: You changed semantic search settings. Consider rebuilding the DB:")
                print("  zotero-mcp update-db --force-rebuild")
            return 0
        else:
            updated_paths = []
            failed_paths = []
            for candidate in config_paths:
                result = update_claude_config(
                    candidate,
                    exe_path,
                    local=use_local,
                    api_key=api_key,
                    library_id=library_id,
                    library_type=library_type,
                    semantic_config=semantic_config,
                    webdav=webdav,
                )
                if result:
                    updated_paths.append(result)
                else:
                    failed_paths.append(candidate)

            if updated_paths:
                print("\nSetup complete!")
                # Always show the absolute path(s) actually written so a
                # mismatch with the running Claude Desktop build is obvious
                # (issue #392).
                if len(updated_paths) == 1:
                    print(f"Claude Desktop config written to: {updated_paths[0].resolve()}")
                else:
                    print("Claude Desktop config written to every detected location:")
                    for path in updated_paths:
                        print(f"  - {path.resolve()}")
                    print("(Multiple Claude Desktop builds were detected, so all of their "
                          "configs were updated.)")
                for path in failed_paths:
                    print(f"Warning: could not write config to: {path}")
                print("To use Zotero in Claude Desktop:")
                print("1. Restart Claude Desktop if it's running")
                print("2. In Claude, type: /tools zotero")
                if semantic_config_changed:
                    print("\nSemantic Search:")
                    print("- Configured with", semantic_config.get("embedding_model", "default"), "embedding model")
                    print("- To change the configuration, run: zotero-mcp setup --semantic-config-only")
                    print("- The config file is located at: ~/.config/zotero-mcp/config.json")
                    print("- You may need to rebuild your database: zotero-mcp update-db --force-rebuild")
                else:
                    print("\nSemantic Search:")
                    print("- To update the database, run: zotero-mcp update-db")
                    print("- Use zotero_semantic_search tool in Claude for AI-powered search")
                if use_local:
                    print("\nNote: Make sure Zotero desktop is running and the local API is enabled in preferences.")
                else:
                    missing = []
                    if not api_key:
                        missing.append("API key")
                    if not library_id:
                        missing.append("Library ID")
                    if missing:
                        print(f"\nWarning: The following required settings for Web API were not provided: {', '.join(missing)}")
                        print("You may need to set these as environment variables or reconfigure.")
                return 0
            else:
                print("\nSetup failed. See errors above.")
                return 1
    except Exception as e:
        print(f"\nSetup failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
