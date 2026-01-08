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
import shutil
import sys
from pathlib import Path


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


def find_claude_config():
    """Find Claude Desktop config file path."""
    config_paths = []

    # macOS
    if sys.platform == "darwin":
        # Try both old and new paths
        config_paths.append(Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
        config_paths.append(Path.home() / "Library" / "Application Support" / "Claude Desktop" / "claude_desktop_config.json")

    # Windows
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            config_paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
            config_paths.append(Path(appdata) / "Claude Desktop" / "claude_desktop_config.json")

    # Linux
    else:
        config_home = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        config_paths.append(Path(config_home) / "Claude" / "claude_desktop_config.json")
        config_paths.append(Path(config_home) / "Claude Desktop" / "claude_desktop_config.json")

    # Check all possible locations
    for path in config_paths:
        if path.exists():
            print(f"Found Claude Desktop config at: {path}")
            return path

    # Return the default path for the platform if not found
    # We'll use the newer "Claude Desktop" path as default
    if sys.platform == "darwin":  # macOS
        default_path = Path.home() / "Library" / "Application Support" / "Claude Desktop" / "claude_desktop_config.json"
    elif sys.platform == "win32":  # Windows
        appdata = os.environ.get("APPDATA", "")
        default_path = Path(appdata) / "Claude Desktop" / "claude_desktop_config.json"
    else:  # Linux and others
        config_home = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        default_path = Path(config_home) / "Claude Desktop" / "claude_desktop_config.json"

    print(f"Claude Desktop config not found. Using default path: {default_path}")
    return default_path

def setup_semantic_search(existing_semantic_config: dict = None, semantic_config_only_arg: bool = False) -> dict:
    """Interactive setup for semantic search configuration."""
    print("\n=== Semantic Search Configuration ===")

    if existing_semantic_config:
        # Display config without sensitive info
        model = existing_semantic_config.get("embedding_model", "unknown")
        name = existing_semantic_config.get("embedding_config", {}).get("model_name", "unknown")
        update_freq = existing_semantic_config.get("update_config", {}).get("update_frequency", "unknown")
        db_path = existing_semantic_config.get("zotero_db_path", "auto-detect")
        print("Found existing semantic search configuration:")
        print(f"  - Embedding model: {model}")
        print(f"  - Embedding model name: {name}")
        print(f"  - Update frequency: {update_freq}")
        print(f"  - Zotero database path: {db_path}")
        print("You can keep it or change it.")
        print("If you change to a new configuration, a database rebuild is advised.")
        print("Would you like to keep your existing configuration? (y/n): ", end="")
        if input().strip().lower() in ['y', 'yes']:
            return existing_semantic_config

    print("Configure embedding models for semantic search over your Zotero library.")

    # Choose embedding model
    print("\nAvailable embedding models:")
    print("1. Default (all-MiniLM-L6-v2) - Free, runs locally")
    print("2. OpenAI - Better quality, requires API key")
    print("3. Gemini - Better quality, requires API key")

    while True:
        choice = input("\nChoose embedding model (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3")

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

        while True:
            model_choice = input("Choose OpenAI model (1-2): ").strip()
            if model_choice in ["1", "2"]:
                break
            print("Please enter 1 or 2")

        if model_choice == "1":
            config["embedding_config"] = {"model_name": "text-embedding-3-small"}
        else:
            config["embedding_config"] = {"model_name": "text-embedding-3-large"}

        # Get API key
        api_key = getpass.getpass("Enter your OpenAI API key (hidden): ").strip()
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            print("Warning: No API key provided. Set OPENAI_API_KEY environment variable.")

        # Get optional base URL
        base_url = input("Enter custom OpenAI base URL (leave blank for default): ").strip()
        if base_url:
            config["embedding_config"]["base_url"] = base_url
            print(f"Using custom OpenAI base URL: {base_url}")
        else:
            print("Using default OpenAI base URL")

    elif choice == "3":
        config["embedding_model"] = "gemini"

        # Choose Gemini model
        print("\nGemini embedding models:")
        print("1. models/text-embedding-004 (recommended)")
        print("2. models/gemini-embedding-exp-03-07 (experimental)")

        while True:
            model_choice = input("Choose Gemini model (1-2): ").strip()
            if model_choice in ["1", "2"]:
                break
            print("Please enter 1 or 2")

        if model_choice == "1":
            config["embedding_config"] = {"model_name": "models/text-embedding-004"}
        else:
            config["embedding_config"] = {"model_name": "models/gemini-embedding-exp-03-07"}

        # Get API key
        api_key = getpass.getpass("Enter your Gemini API key (hidden): ").strip()
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            print("Warning: No API key provided. Set GEMINI_API_KEY environment variable.")

        # Get optional base URL
        base_url = input("Enter custom Gemini base URL (leave blank for default): ").strip()
        if base_url:
            config["embedding_config"]["base_url"] = base_url
            print(f"Using custom Gemini base URL: {base_url}")
        else:
            print("Using default Gemini base URL")

    # Configure update frequency
    print("\n=== Database Update Configuration ===")
    print("Configure how often the semantic search database is updated:")
    print("1. Manual - Update only when you run 'zotero-mcp update-db'")
    print("2. Auto - Automatically update on server startup")
    print("3. Daily - Automatically update once per day")
    print("4. Every N days - Automatically update every N days")

    while True:
        update_choice = input("\nChoose update frequency (1-4): ").strip()
        if update_choice in ["1", "2", "3", "4"]:
            break
        print("Please enter 1, 2, 3, or 4")

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
            "update_frequency": "daily"
        }
        print("Database will be updated once per day.")
    elif update_choice == "4":
        while True:
            try:
                days = int(input("Enter number of days between updates: ").strip())
                if days > 0:
                    break
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

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

    # Configure Zotero database path
    print("\n=== Zotero Database Path ===")
    print("By default, zotero-mcp auto-detects the Zotero database location.")
    print("If Zotero is installed in a custom location, you can specify the path here.")
    default_db_path = existing_semantic_config.get("zotero_db_path", "") if existing_semantic_config else ""
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
    config["extraction"] = {"pdf_max_pages": pdf_max_pages}
    if zotero_db_path:
        config["zotero_db_path"] = zotero_db_path

    return config


def save_semantic_search_config(config: dict, semantic_config_path: Path) -> bool:
    """Save semantic search configuration to file."""
    try:
        # Ensure config directory exists
        semantic_config_dir = semantic_config_path.parent
        semantic_config_dir.mkdir(parents=True, exist_ok=True)

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

        # Write config
        with open(semantic_config_path, 'w') as f:
            json.dump(full_semantic_config, f, indent=2)

        print(f"Semantic search configuration saved to: {semantic_config_path}")
        return True

    except Exception as e:
        print(f"Error saving semantic search config: {e}")
        return False

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


def update_claude_config(config_path, zotero_mcp_path, local=True, api_key=None, library_id=None, library_type="user", semantic_config=None):
    """Update Claude Desktop config to add zotero-mcp."""
    # Create directory if it doesn't exist
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new one
    if config_path.exists():
        try:
            with open(config_path) as f:
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

    # Add API key and library settings for web API
    if not local:
        if api_key:
            env_settings["ZOTERO_API_KEY"] = api_key
        if library_id:
            env_settings["ZOTERO_LIBRARY_ID"] = library_id
        if library_type:
            env_settings["ZOTERO_LIBRARY_TYPE"] = library_type

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

    # Add or update zotero config
    config["mcpServers"]["zotero"] = {
        "command": zotero_mcp_path,
        "env": env_settings
    }

    # Write updated config
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nSuccessfully wrote config to: {config_path}")
    except Exception as e:
        print(f"Error writing config file: {str(e)}")
        return False

    return config_path


def _write_standalone_config(local: bool, api_key: str, library_id: str, library_type: str, semantic_config: dict, no_claude: bool = False) -> Path:
    """Write a central config file used by semantic search and provide client env."""
    cfg_dir = Path.home() / ".config" / "zotero-mcp"
    cfg_dir.mkdir(parents=True, exist_ok=True)
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

    full["client_env"] = client_env

    with open(cfg_path, 'w') as f:
        json.dump(full, f, indent=2)

    return cfg_path


def main(cli_args=None):
    """Main function to run the setup helper."""
    parser = argparse.ArgumentParser(description="Configure zotero-mcp for Claude Desktop")
    parser.add_argument("--no-local", action="store_true", help="Configure for Zotero Web API instead of local API")
    parser.add_argument("--no-claude", action="store_true", help="Don't setup Claude Desktop config: instead store settings in config file.")
    parser.add_argument("--api-key", help="Zotero API key (only needed with --no-local)")
    parser.add_argument("--library-id", help="Zotero library ID (only needed with --no-local)")
    parser.add_argument("--library-type", choices=["user", "group"], default="user",
                        help="Zotero library type (only needed with --no-local)")
    parser.add_argument("--config-path", help="Path to Claude Desktop config file")
    parser.add_argument("--skip-semantic-search", action="store_true",
                        help="Skip semantic search configuration")
    parser.add_argument("--semantic-config-only", action="store_true",
                        help="Only configure semantic search, skip Zotero setup")

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
    semantic_config_changed = False

    # Handle semantic search only configuration
    if args.semantic_config_only:
        print("Configuring semantic search only...")
        new_semantic_config = setup_semantic_search(existing_semantic_config)
        semantic_config_changed = existing_semantic_config != new_semantic_config
        # only save if semantic config changed
        if semantic_config_changed:
            if save_semantic_search_config(new_semantic_config, semantic_config_path):
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

    # Find Claude Desktop config unless --no-claude
    config_path = None
    if not args.no_claude:
        config_path = args.config_path
        if not config_path:
            config_path = find_claude_config()
        else:
            print(f"Using specified config path: {config_path}")
            config_path = Path(config_path)
        if not config_path:
            print("Error: Could not determine Claude Desktop config path.")
            return 1

    # Update config
    use_local = not args.no_local
    api_key = args.api_key
    library_id = args.library_id
    library_type = args.library_type

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
            new_semantic_config = setup_semantic_search(existing_semantic_config)
            if existing_semantic_config != new_semantic_config:
                semantic_config_changed = True
                existing_semantic_config = new_semantic_config  # Update the config to use
                save_semantic_search_config(existing_semantic_config, semantic_config_path)

    print("\nSetup with the following settings:")
    print(f"  Local API: {use_local}")
    if not use_local:
        print(f"  API Key: {api_key or 'Not provided'}")
        print(f"  Library ID: {library_id or 'Not provided'}")
        print(f"  Library Type: {library_type}")

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
                no_claude=args.no_claude
            )
            print("\nSetup complete (standalone/web mode)!")
            print(f"Config saved to: {cfg_path}")
            # Emit one-line client_env for easy copy/paste
            try:
                with open(cfg_path) as f:
                    full = json.load(f)
                env_line = json.dumps(full.get("client_env", {}), separators=(',', ':'))
                print("Client environment (single-line JSON):")
                print(env_line)
            except Exception:
                pass
            if semantic_config_changed:
                print("\nNote: You changed semantic search settings. Consider rebuilding the DB:")
                print("  zotero-mcp update-db --force-rebuild")
            return 0
        else:
            updated_config_path = update_claude_config(
                config_path,
                exe_path,
                local=use_local,
                api_key=api_key,
                library_id=library_id,
                library_type=library_type,
                semantic_config=semantic_config
            )
            if updated_config_path:
                print("\nSetup complete!")
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