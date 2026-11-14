#!/usr/bin/env bash
# Live check for the annotation workflow, against a REAL Zotero. Run it by hand.
#
# What it does, on one TEST attachment:
#   1. creates three purple area boxes: one on a normal page, one on a ROTATED
#      page, one on a third page;
#   2. lists them with `annotations list --positions` and prints the stored
#      position of each (page index, rects in PDF points, and the normalized rect);
#   3. moves the rotated-page box with `annotations update --rect`;
#   4. draws what Zotero stores with `annotations preview` (before and after the move);
#   5. PAUSES so you can look at the boxes in the Zotero reader, then
#   6. PERMANENTLY deletes the three boxes (also if anything fails along the way).
#
# It writes to your library, and your library syncs to zotero.org and WebDAV, so:
#   * PAUSE SYNC IN ZOTERO FIRST (Settings > Sync, or the sync button's menu).
#   * Use a throwaway attachment, not one you care about.
#   * Writes in local mode need `zotero-mcp authorize-local` once.
#
# Usage:
#   I_PAUSED_SYNC=yes ATTACH=ABCD1234 ROT_PAGE=2 ./scripts/live_check_annotations.sh
#
#   ATTACH     key of the PDF ATTACHMENT (not the parent item); `zotero-cli get children ITEM_KEY`
#   ROT_PAGE   a page of that PDF that has /Rotate set (a landscape page); default 2
#   PLAIN_PAGE a normal page for the first box (default 1); THIRD_PAGE for the third (default 3)
#   OUT        where the PNGs go (default: a new temp folder)
#   ZCLI       the command to run (default: zotero-cli)
#
# What to look at afterwards (it prints this too):
#   * $OUT/before/*.png: each purple box sits on the spot named in its label, the one
#     on the rotated page included. A box that is shifted here is shifted in Zotero.
#   * $OUT/after/*.png: only the rotated-page box moved; the other two are unchanged.
#   * The Zotero reader at the pause: boxes line up with the PNGs.
#   * The stored positions printed in step 2: `rect` equals the rect the box was created with.
set -euo pipefail

ZCLI="${ZCLI:-zotero-cli}"
ATTACH="${ATTACH:?set ATTACH to the test PDF attachment key}"
PLAIN_PAGE="${PLAIN_PAGE:-1}"
ROT_PAGE="${ROT_PAGE:-2}"
THIRD_PAGE="${THIRD_PAGE:-3}"
OUT="${OUT:-$(mktemp -d -t zotero_live_check)}"

if [[ "${I_PAUSED_SYNC:-}" != "yes" ]]; then
  echo "This writes to your real Zotero library. Pause sync first, then re-run with I_PAUSED_SYNC=yes." >&2
  exit 2
fi

KEYS=()

cleanup() {
  # Permanent delete: a trashed annotation still draws in the reader.
  local status=$?
  if ((${#KEYS[@]})); then
    echo
    echo "Deleting the ${#KEYS[@]} test annotation(s) permanently..."
    for key in "${KEYS[@]}"; do
      "$ZCLI" annotations delete "$key" || echo "  could not delete $key: remove it by hand" >&2
    done
  fi
  exit "$status"
}
trap cleanup EXIT

# `annotations create` prints "**Annotation Key:** KEY". Usage: create_box VAR page rect comment.
# Sets VAR to the key in this shell (not a subshell), so the cleanup trap knows about it.
create_box() {
  local __var="$1" page="$2" rect="$3" comment="$4" output key
  output="$("$ZCLI" annotations create --attachment-key "$ATTACH" --page "$page" \
              --rect "$rect" --comment "$comment" --color purple)"
  key="$(printf '%s\n' "$output" | sed -n 's/.*Annotation Key:\*\* \([A-Z0-9]*\).*/\1/p' | head -n 1)"
  [[ -n "$key" ]] || { echo "could not read the new annotation's key from:" >&2; echo "$output" >&2; exit 1; }
  KEYS+=("$key")
  printf -v "$__var" '%s' "$key"
}

echo "== 1. Creating three boxes on $ATTACH"
RECT_A="0.10,0.30,0.30,0.15"
RECT_B="0.55,0.40,0.30,0.20"
RECT_C="0.30,0.60,0.25,0.20"
create_box KEY_A "$PLAIN_PAGE" "$RECT_A" "LIVE-CHECK A plain page"
create_box KEY_B "$ROT_PAGE" "$RECT_B" "LIVE-CHECK B rotated page"
create_box KEY_C "$THIRD_PAGE" "$RECT_C" "LIVE-CHECK C third page"
echo "   A=$KEY_A (page $PLAIN_PAGE, rect $RECT_A)"
echo "   B=$KEY_B (page $ROT_PAGE, rect $RECT_B)  <- rotated"
echo "   C=$KEY_C (page $THIRD_PAGE, rect $RECT_C)"

echo
echo "== 2. Stored positions (annotations list --positions)"
echo "   Each normalized 'rect' should equal the rect given above."
"$ZCLI" --json annotations list --item-key "$ATTACH" --positions --limit 500 \
  | python3 -c '
import json, sys
mine = {"'"$KEY_A"'", "'"$KEY_B"'", "'"$KEY_C"'"}
for record in json.load(sys.stdin)["data"]:
    if record["annotation_key"] in mine:
        position = record["position"]
        print("  ", record["annotation_key"], "page_index", position["page_index"],
              "rect", position["rect"], "rects(pt)", position["rects"])
'

echo
echo "== 3. Previewing what Zotero stores (before the move)"
"$ZCLI" annotations preview --attachment-key "$ATTACH" --pages "$PLAIN_PAGE,$ROT_PAGE,$THIRD_PAGE" --out "$OUT/before"

echo
echo "== 4. Moving box B (rotated page) with update --rect"
NEW_RECT_B="0.50,0.10,0.20,0.10"
"$ZCLI" annotations update "$KEY_B" --rect "$NEW_RECT_B"
"$ZCLI" annotations preview --attachment-key "$ATTACH" --pages "$PLAIN_PAGE,$ROT_PAGE,$THIRD_PAGE" --out "$OUT/after"

echo
echo "== What to look at =="
echo "   Before: $OUT/before   After: $OUT/after"
echo "   - before/: boxes A, B, C sit where created (B on the rotated page too)."
echo "   - after/:  only B moved, to rect $NEW_RECT_B; A and C are unchanged."
echo "   - the Zotero reader: open page $ROT_PAGE with"
echo "       $ZCLI open $ATTACH --page $ROT_PAGE"
echo "     and check B matches after/*.png, and that A and C match before/*.png."
echo
read -r -p "Press Enter when you have looked; the three test boxes are then deleted permanently. "
