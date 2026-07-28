#!/bin/sh
set -eu

app_mode="${ZOTERO_APP:-server}"

if [ "$app_mode" = "cli" ]; then
    if [ "$#" -eq 0 ]; then
        set -- --help
    fi
    exec zotero-cli "$@"
fi

if [ "$#" -eq 0 ]; then
    set -- serve --transport "${ZOTERO_TRANSPORT:-stdio}"
fi

exec zotero-mcp "$@"
