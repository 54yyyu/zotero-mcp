#!/bin/bash
# Sync Zotero MCP semantic search database to sompaserv
#
# Run this after updating the database with:
#   zotero-mcp update-db --fulltext
#
# On sompaserv, run the server in read-only mode:
#   zotero-mcp serve --read-only

set -e

DB_DIR="$HOME/.config/zotero-mcp/chroma_db"
REMOTE_HOST="sompaserv"
REMOTE_DIR="~/.config/zotero-mcp/chroma_db"

if [ ! -d "$DB_DIR" ]; then
    echo "Error: Database directory not found: $DB_DIR"
    echo "Run 'zotero-mcp update-db' first to build the database."
    exit 1
fi

echo "Syncing Zotero MCP database to $REMOTE_HOST..."
rsync -avz --delete "$DB_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "Done. Database synced to $REMOTE_HOST:$REMOTE_DIR"
