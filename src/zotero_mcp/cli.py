"""
Command-line interface for Zotero MCP server.
"""

import argparse
import sys

from zotero_mcp.server import mcp


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
        choices=["stdio", "sse"],
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
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Print version information")
    
    args = parser.parse_args()
    
    # If no command is provided, default to 'serve'
    if not args.command:
        args.command = "serve"
    
    if args.command == "version":
        from zotero_mcp._version import __version__
        print(f"Zotero MCP v{__version__}")
        sys.exit(0)
    
    elif args.command == "setup":
        from zotero_mcp.setup_helper import main as setup_main
        sys.exit(setup_main())
    
    elif args.command == "serve":
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        elif args.transport == "sse":
            mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
