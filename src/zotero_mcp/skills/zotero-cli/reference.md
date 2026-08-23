# zotero-cli command reference

Generated from the CLI's own argument parser by
`scripts/gen_skill_reference.py` -- do not edit by hand.

Every command also accepts `--json` (machine-readable envelope on stdout) and
`-v` (diagnostics on stderr), before or after the command name. Run
`zotero-cli --json-schema` for the output contract.


## `config`

 - `--show-secrets` -- Show full API keys

## `search (alias: s)`

 - `<query>` -- Search query
 - `--mode` -- one of `items`, `tag`, `citekey`, `advanced`, `semantic`, `notes` -- default `items` -- Search mode (default: items)
 - `--qmode` -- one of `titleCreatorYear`, `everything` -- default `titleCreatorYear`
 - `--collection` -- Scope to a collection key
 - `--limit` -- default `10`
 - `--conditions` -- JSON conditions for advanced mode
 - `--join-mode` -- one of `all`, `any` -- default `all`
 - `--sort-by`
 - `--sort-direction` -- one of `asc`, `desc` -- default `asc`
 - `--filters` -- JSON filters for semantic mode
 - `--detail` -- one of `keys_only`, `summary`, `full` -- default `summary` -- How much of each item --json returns (no effect on markdown output)

## `get (alias: g)`

### `get metadata`

 - `<item_key>`
 - `--no-abstract`
 - `--output-format` -- one of `markdown`, `bibtex` -- default `markdown`

### `get fulltext`

 - `<item_key>`

### `get bibtex`

 - `<item_key>`

### `get collections`

 - `--limit` -- default `500`

### `get collection-items`

 - `<collection_key>`
 - `--detail` -- one of `keys_only`, `summary`, `full` -- default `summary`
 - `--limit` -- default `50`
 - `--offset` -- Index of the first item to return, for paging a collection larger than --limit

### `get children`

 - `<item_key>`
 - `--item-keys` -- Comma-separated keys for batch mode

### `get tags`

 - `--limit` -- default `500`

### `get recent`

 - `--limit` -- default `10`
 - `--collection`

### `get libraries`

### `get feeds`

### `get feed-items`

 - `<library_id>`
 - `--limit` -- default `20`

## `annotations (alias: ann)`

### `annotations list`

 - `--item-key`
 - `--pdf-extraction`
 - `--limit` -- default `100`
 - `--format` -- one of `markdown`, `json` -- default `markdown`

### `annotations update`

 - `<annotation_key>`
 - `--text`
 - `--comment`
 - `--color`
 - `--add-tags` -- Comma-separated tags to add
 - `--remove-tags` -- Comma-separated tags to remove

### `annotations delete`

 - `<annotation_key>`

### `annotations create`

 - `--attachment-key` -- **required**
 - `--page` -- **required**
 - `--text` -- **required**
 - `--comment`
 - `--color` -- default `#ffd400`

## `notes (alias: n)`

### `notes list`

 - `--item-key`
 - `--limit` -- default `20`
 - `--full`
 - `--raw-html`

### `notes create`

 - `--item-key` -- **required**
 - `--title`
 - `--text` -- Note text (use - to read from stdin)
 - `--tags`

### `notes update`

 - `--item-key` -- **required**
 - `--text` -- New text (use - for stdin)

### `notes delete`

 - `--item-key` -- **required**

## `add`

### `add doi`

 - `<doi>`
 - `--collections` -- Comma-separated collection keys, names, or paths
 - `-c, --collection` -- Collection key, name, or parent/child path (repeatable; not comma-split, so names with commas work)
 - `--tags` -- Comma-separated tags
 - `--if-exists` -- one of `file`, `skip`, `duplicate` -- default `file` -- When the item already exists: 'file' (default) reuses it and adds missing collections/tags; 'skip' leaves it untouched; 'duplicate' creates a new item anyway
 - `--create-collections` -- Create collections that don't exist yet (including parent/child paths)
 - `--attach-mode` -- one of `auto`, `linked_url`, `import_file`, `none`, `required` -- default `auto`

### `add url`

 - `<url>`
 - `--collections` -- Comma-separated collection keys, names, or paths
 - `-c, --collection` -- Collection key, name, or parent/child path (repeatable; not comma-split, so names with commas work)
 - `--tags` -- Comma-separated tags
 - `--if-exists` -- one of `file`, `skip`, `duplicate` -- default `file` -- When the item already exists: 'file' (default) reuses it and adds missing collections/tags; 'skip' leaves it untouched; 'duplicate' creates a new item anyway
 - `--create-collections` -- Create collections that don't exist yet (including parent/child paths)
 - `--attach-mode` -- one of `auto`, `linked_url`, `import_file`, `none`, `required` -- default `auto`

### `add file`

 - `--filepath` -- **required**
 - `--title` -- Override title if metadata extraction misses
 - `--item-type` -- default `document` -- Zotero item type for the new item (default: document)
 - `--collections` -- Comma-separated collection keys, names, or paths
 - `-c, --collection` -- Collection key, name, or parent/child path (repeatable; not comma-split, so names with commas work)
 - `--tags` -- Comma-separated tags
 - `--if-exists` -- one of `file`, `skip`, `duplicate` -- default `file` -- When the item already exists: 'file' (default) reuses it and adds missing collections/tags; 'skip' leaves it untouched; 'duplicate' creates a new item anyway
 - `--create-collections` -- Create collections that don't exist yet (including parent/child paths)

### `add isbn`

 - `<isbn>`
 - `--collections` -- Comma-separated collection keys, names, or paths
 - `-c, --collection` -- Collection key, name, or parent/child path (repeatable; not comma-split, so names with commas work)
 - `--tags` -- Comma-separated tags
 - `--if-exists` -- one of `file`, `skip`, `duplicate` -- default `file` -- When the item already exists: 'file' (default) reuses it and adds missing collections/tags; 'skip' leaves it untouched; 'duplicate' creates a new item anyway
 - `--create-collections` -- Create collections that don't exist yet (including parent/child paths)

### `add bibtex`

 - `--bibtex` -- Inline BibTeX (use - to read from stdin)
 - `--file` -- Path to a .bib/.bibtex file
 - `--collections` -- Comma-separated collection keys, names, or paths
 - `-c, --collection` -- Collection key, name, or parent/child path (repeatable; not comma-split, so names with commas work)
 - `--tags` -- Comma-separated tags
 - `--if-exists` -- one of `file`, `skip`, `duplicate` -- default `file` -- When the item already exists: 'file' (default) reuses it and adds missing collections/tags; 'skip' leaves it untouched; 'duplicate' creates a new item anyway
 - `--create-collections` -- Create collections that don't exist yet (including parent/child paths)
 - `--attach-mode` -- one of `auto`, `linked_url`, `import_file`, `none`, `required` -- default `auto`

### `add csl-json`

 - `--json` -- Inline CSL JSON (use - to read from stdin)
 - `--file` -- Path to a .json/.csljson file
 - `--collections` -- Comma-separated collection keys, names, or paths
 - `-c, --collection` -- Collection key, name, or parent/child path (repeatable; not comma-split, so names with commas work)
 - `--tags` -- Comma-separated tags
 - `--if-exists` -- one of `file`, `skip`, `duplicate` -- default `file` -- When the item already exists: 'file' (default) reuses it and adds missing collections/tags; 'skip' leaves it untouched; 'duplicate' creates a new item anyway
 - `--create-collections` -- Create collections that don't exist yet (including parent/child paths)
 - `--attach-mode` -- one of `auto`, `linked_url`, `import_file`, `none`, `required` -- default `auto`

## `collections (alias: coll)`

### `collections create`

 - `<name>`
 - `--parent`

### `collections search`

 - `<query>`

### `collections manage`

 - `--item-keys` -- **required**
 - `--add-to`
 - `--remove-from`

## `tags`

 - `--query`
 - `--tag`
 - `--add`
 - `--remove`
 - `--limit` -- default `50`

## `edit`

 - `<item_key>`
 - `--title`
 - `--creators` -- JSON array of creators
 - `--date`
 - `--publication-title`
 - `--abstract`
 - `--tags` -- Replace all tags (comma-separated)
 - `--add-tags`
 - `--remove-tags`
 - `--collections` -- Add to collections (comma-separated keys)
 - `--collection-names` -- Add to collections (comma-separated names)
 - `--doi`
 - `--url`
 - `--extra`
 - `--volume`
 - `--issue`
 - `--pages`
 - `--publisher`
 - `--issn`
 - `--language`
 - `--short-title`
 - `--edition`
 - `--isbn`
 - `--book-title`

## `duplicates`

### `duplicates find`

 - `--method` -- one of `title`, `doi`, `both` -- default `both`
 - `--collection`
 - `--limit` -- default `50`

### `duplicates merge`

 - `--keeper-key` -- **required**
 - `--duplicate-keys` -- **required**
 - `--dry-run`

## `db`

### `db update`

 - `--force-rebuild`
 - `--limit`
 - `--fulltext`
 - `--allow-mass-deletion`
 - `--config-path`
 - `--db-path`
 - `--openai-batch`
 - `--no-openai-batch`

### `db batch-status`

 - `--batch-id`
 - `--config-path`

### `db batch-import`

 - `--batch-id`
 - `--config-path`

### `db status`

 - `--config-path`

### `db inspect`

 - `--limit` -- default `20`
 - `--filter-text`
 - `--show-documents`
 - `--stats`
 - `--config-path`

## `library`

 - `<action>` -- one of `switch`, `list`, `reset`
 - `--library-id`
 - `--library-type` -- one of `user`, `group` -- default `group`

## `outline`

 - `<item_key>`

## `read`

 - `<item_key>`
 - `--start-page` -- **required**
 - `--end-page` -- Defaults to --start-page (a single page)

## `attach`

 - `<item_key>`
 - `--file` -- Path to a local file to upload
 - `--url` -- URL to attach as a link
 - `--filename` -- Override the stored filename

## `delete`

### `delete item`

 - `<item_key>`
 - `--allow-note` -- Permit deleting a note (refused otherwise, since a note is usually deleted by mistake)

### `delete collection`

 - `<collection_key>`

### `delete annotation`

 - `<annotation_key>`

## `export`

 - `--item-keys` -- Comma-separated item keys
 - `--collection` -- Export a whole collection instead
 - `--style` -- default `apa` -- CSL style (default: apa)
 - `--format` -- one of `bib`, `citation`, `bibtex` -- default `bib`

## `related`

 - `<identifier>` -- DOI, arXiv ID, or Zotero item key
 - `--direction` -- one of `references`, `citations`, `both` -- default `both`
 - `--limit` -- default `20`

## `coverage`

 - `--collection` -- Scope to one collection
 - `--limit` -- default `200`

## `synthesize`

 - `--collection`
 - `--tag` -- Comma-separated tags to scope by
 - `--limit` -- default `200`
 - `--format` -- one of `markdown`, `json` -- default `markdown`

## `path`

 - `<item_key>`

## `batch`

 - `--item-keys` -- Comma-separated item keys
 - `--query` -- Select items by search query instead
 - `--tag` -- Comma-separated tags to select by
 - `--add-tags` -- Comma-separated tags to add
 - `--remove-tags` -- Comma-separated tags to remove
 - `--set` -- JSON object of Extra keys to set
 - `--remove-keys` -- Comma-separated Extra keys to remove
 - `--limit` -- default `50`
