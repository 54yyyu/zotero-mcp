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

from zotero_mcp.server import mcp


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
    sensitive_keys = ["ZOTERO_API_KEY", "ZOTERO_LIBRARY_ID", "API_KEY", "LIBRARY_ID"]
    
    for key in sensitive_keys:
        if key in obfuscated:
            obfuscated[key] = obfuscate_sensitive_value(obfuscated[key])
    
    return obfuscated


def load_claude_desktop_env_vars():
    """Load Zotero environment variables from Claude Desktop config."""
    from zotero_mcp.setup_helper import find_claude_config
    
    try:
        config_path = find_claude_config()
        if not config_path or not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract Zotero MCP server environment variables
        mcp_servers = config.get("mcpServers", {})
        zotero_config = mcp_servers.get("zotero", {})
        env_vars = zotero_config.get("env", {})
        
        return env_vars
    
    except Exception:
        return {}


def apply_environment_variables(env_vars):
    """Apply environment variables to current process."""
    for key, value in env_vars.items():
        if key not in os.environ:  # Don't override existing env vars
            os.environ[key] = str(value)


def setup_zotero_environment():
    """Setup Zotero environment for CLI commands."""
    # Load environment variables from Claude Desktop config
    claude_env_vars = load_claude_desktop_env_vars()
    
    # Apply fallback defaults for local Zotero if no config found
    fallback_env_vars = {
        "ZOTERO_LOCAL": "true",
        "ZOTERO_LIBRARY_ID": "0",
    }
    
    # Apply Claude Desktop env vars first
    apply_environment_variables(claude_env_vars)
    
    # Apply fallbacks only if not already set
    apply_environment_variables(fallback_env_vars)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Zotero Model Context Protocol server"
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
    setup_parser = subparsers.add_parser("setup", help="Configure zotero-mcp for Claude Desktop")
    setup_parser.add_argument("--no-local", action="store_true", 
                             help="Configure for Zotero Web API instead of local API")
    setup_parser.add_argument("--api-key", help="Zotero API key (only needed with --no-local)")
    setup_parser.add_argument("--library-id", help="Zotero library ID (only needed with --no-local)")
    setup_parser.add_argument("--library-type", choices=["user", "group"], default="user", 
                             help="Zotero library type (only needed with --no-local)")
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
    update_db_parser.add_argument("--config-path", 
                                 help="Path to semantic search configuration file")
    
    # Database status command
    db_status_parser = subparsers.add_parser("db-status", help="Show semantic search database status")
    db_status_parser.add_argument("--config-path",
                                 help="Path to semantic search configuration file")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update zotero-mcp to the latest version")
    update_parser.add_argument("--check-only", action="store_true",
                              help="Only check for updates without installing")
    update_parser.add_argument("--force", action="store_true",
                              help="Force update even if already up to date")
    update_parser.add_argument("--method", choices=["pip", "uv", "conda", "pipx"],
                              help="Override auto-detected installation method")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Print version information")
    
    # Setup info command
    setup_info_parser = subparsers.add_parser("setup-info", help="Show installation path and configuration info for MCP clients")
    
    args = parser.parse_args()
    
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
        
        # Load current environment configuration
        claude_env_vars = load_claude_desktop_env_vars()
        
        # If no Claude config found, use defaults
        if not claude_env_vars:
            claude_env_vars = {"ZOTERO_LOCAL": "true"}
        
        print("=== Zotero MCP Setup Information ===")
        print()
        print("🔧 Installation Details:")
        print(f"  Command path: {executable_path}")
        print(f"  Python path: {sys.executable}")
        
        # Detect installation method
        try:
            # Check if installed via uv
            result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, timeout=5)
            if "zotero-mcp" in result.stdout:
                print("  Installation method: uv tool")
            else:
                # Check pip
                result = subprocess.run([sys.executable, "-m", "pip", "show", "zotero-mcp"], 
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
        obfuscated_env_vars = obfuscate_config_for_display(claude_env_vars)
        print(f"  Environment (single-line): {json.dumps(obfuscated_env_vars, separators=(',', ':'))}")
        print("  💡 Note: This shows client config. Shell variables may override for CLI use.")
        
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
                print(f"  Auto update: {update_config.get('auto_update', False)}")
                print(f"  Update frequency: {update_config.get('update_frequency', 'manual')}")
                print(f"  Last update: {update_config.get('last_update', 'Never')}")
                print(f"  Should update: {status.get('should_update', False)}")
                
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
        config_path = args.config_path
        if not config_path:
            config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        else:
            config_path = Path(config_path)
        
        print(f"Using configuration: {config_path}")
        
        try:
            # Create semantic search instance
            search = create_semantic_search(str(config_path))
            
            print("Starting database update...")
            stats = search.update_database(
                force_full_rebuild=args.force_rebuild,
                limit=args.limit
            )
            
            print(f"\nDatabase update completed:")
            print(f"- Total items: {stats.get('total_items', 0)}")
            print(f"- Processed: {stats.get('processed_items', 0)}")
            print(f"- Added: {stats.get('added_items', 0)}")
            print(f"- Updated: {stats.get('updated_items', 0)}")
            print(f"- Skipped: {stats.get('skipped_items', 0)}")
            print(f"- Errors: {stats.get('errors', 0)}")
            print(f"- Duration: {stats.get('duration', 'Unknown')}")
            
            if stats.get('error'):
                print(f"Error: {stats['error']}")
                sys.exit(1)
            
        except Exception as e:
            print(f"Error updating database: {e}")
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
            print(f"\nUpdate configuration:")
            print(f"- Auto update: {update_config.get('auto_update', False)}")
            print(f"- Frequency: {update_config.get('update_frequency', 'manual')}")
            print(f"- Last update: {update_config.get('last_update', 'Never')}")
            print(f"- Should update: {status.get('should_update', False)}")
            
            if collection_info.get('error'):
                print(f"\nError: {collection_info['error']}")
            
        except Exception as e:
            print(f"Error getting database status: {e}")
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
        # Get transport with a default value if not specified
        transport = getattr(args, "transport", "stdio")
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