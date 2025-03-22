#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup helper for zotero-mcp.

This script provides utilities to automatically configure zotero-mcp
by finding the installed executable and updating Claude Desktop's config.
"""

import argparse
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
    
    for path in potential_paths:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    
    return None


def find_claude_config():
    """Find Claude Desktop config file path."""
    config_paths = []
    
    # macOS
    config_paths.append(Path.home() / "Library" / "Application Support" / "Claude Desktop" / "claude_desktop_config.json")
    
    # Windows
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            config_paths.append(Path(appdata) / "Claude Desktop" / "claude_desktop_config.json")
    
    # Linux
    config_paths.append(Path.home() / ".config" / "Claude Desktop" / "claude_desktop_config.json")
    
    for path in config_paths:
        if path.exists():
            return path
    
    # Return the default path for the platform if not found
    if sys.platform == "darwin":  # macOS
        return config_paths[0]
    elif sys.platform == "win32":  # Windows
        return config_paths[1]
    else:  # Linux and others
        return config_paths[2]


def update_claude_config(config_path, zotero_mcp_path, local=True):
    """Update Claude Desktop config to add zotero-mcp."""
    # Create directory if it doesn't exist
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create new one
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Config file at {config_path} is not valid JSON. Creating new config.")
            config = {}
    else:
        config = {}
    
    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    # Add or update zotero config
    config["mcpServers"]["zotero"] = {
        "command": zotero_mcp_path,
        "env": {
            "ZOTERO_LOCAL": "true" if local else "false"
        }
    }
    
    # Write updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def main():
    """Main function to run the setup helper."""
    parser = argparse.ArgumentParser(description="Configure zotero-mcp for Claude Desktop")
    parser.add_argument("--no-local", action="store_true", help="Configure for Zotero Web API instead of local API")
    parser.add_argument("--api-key", help="Zotero API key (only needed with --no-local)")
    parser.add_argument("--library-id", help="Zotero library ID (only needed with --no-local)")
    parser.add_argument("--library-type", choices=["user", "group"], default="user", 
                        help="Zotero library type (only needed with --no-local)")
    parser.add_argument("--config-path", help="Path to Claude Desktop config file")
    
    args = parser.parse_args()
    
    # Find zotero-mcp executable
    exe_path = find_executable()
    if not exe_path:
        print("Error: Could not find zotero-mcp executable.")
        print("Make sure zotero-mcp is installed and in your PATH.")
        return 1
    print(f"Found zotero-mcp at: {exe_path}")
    
    # Find Claude Desktop config
    config_path = args.config_path
    if not config_path:
        config_path = find_claude_config()
    else:
        config_path = Path(config_path)
    print(f"Using config at: {config_path}")
    
    # Update config
    use_local = not args.no_local
    updated_config_path = update_claude_config(config_path, exe_path, local=use_local)
    
    print(f"Configuration updated at: {updated_config_path}")
    
    # Additional setup for web API
    if not use_local:
        if not args.api_key:
            print("Warning: Using Web API requires a Zotero API key.")
            print("Add ZOTERO_API_KEY to your environment variables.")
        
        if not args.library_id:
            print("Warning: Using Web API requires a Zotero library ID.")
            print("Add ZOTERO_LIBRARY_ID to your environment variables.")
    
    print("\nSetup complete!")
    print("\nTo use Zotero in Claude Desktop:")
    print("1. Restart Claude Desktop if it's running")
    print("2. In Claude, type: /tools zotero")
    
    if use_local:
        print("\nNote: Make sure Zotero desktop is running and the local API is enabled in preferences.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
