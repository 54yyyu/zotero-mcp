// Gecko / Zotero chrome-scope globals. The bootstrap hands these to plugin.js; they are not typed upstream.
declare const Zotero: any;
declare const Services: any;
declare const ChromeUtils: any;
declare const IOUtils: any;
declare const PathUtils: any;
declare const Components: any;
declare const Cc: any;
declare const Ci: any;
/** Build flag: true in `build.mjs --dev` (the harness), false in the released plugin. */
declare const __TEST_HOOKS__: boolean;
