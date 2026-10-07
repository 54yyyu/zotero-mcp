---
name: annotate-paper
description: Read the open paper and write study annotations into its PDF with zotero-cli - a context box on the title, a four-part summary on the abstract, role-coded abstract highlights, one box per figure panel with a teaching comment, colour-coded body highlights, and a reading note. Use when asked to "annotate", "read and annotate", or "help me understand" a paper.
---

# Annotate a paper for understanding

The goal is a PDF someone can study from cold: the background, the argument and every figure explained in place.

## Colour scheme (change it here)

Use exactly these colours so every paper reads the same way. Edit the table to make the scheme your own; the rest
of the skill refers to the roles, not the colours.

| Annotation | Where | Colour | Content |
|---|---|---|---|
| Context box | around the title and authors, page 1 | orange | What a newcomer needs first: the field, key terms defined plainly, the problem's history, why this approach. Start with `CONTEXT`. |
| Four-part summary | around the abstract | magenta | `THE WHOLE PAPER IN FOUR PARTS.` then `// PROBLEM:`, `// METHOD:` (with the real numbers), `// RESULTS:` (numbered), `// TAKEAWAY:` and the limits. |
| Abstract highlights | sentences in the abstract | by role | Prefix each comment with its role: `PROBLEM.` / `GOAL.` / `CONCLUSION.` (yellow), `METHOD.` (blue), `DISCOVERY.` (green). |
| Figure boxes | one per sub-panel | purple | See "Figure boxes" below. |
| Body highlights | key sentences throughout | yellow / blue / green / red | yellow = claim or big idea; blue = method, data or design choice; green = finding or evidence; red = limitation, caveat or critique. |

This skill sits on top of the `zotero-cli` skill. Run `zotero-cli config` once first; writing annotations needs local
write access.

## 1. Find the paper and its PDF

Use the paper that is open in the reader, or the one the user names:

```bash
zotero-cli --json search "title words" --limit 5
zotero-cli --json get children ITEM_KEY          # the PDF attachment's key; supplementary PDFs are attachments too
```

Annotate the main PDF, not the supplement, unless asked.

## 2. Read the whole paper, then look at every figure

```bash
zotero-cli read ITEM_KEY --start-page 1 --end-page 99
```

Read all of it, including Methods and figure captions. Extracted text is unreliable for figures, tables and math, so
render every figure page and look at it yourself:

```bash
zotero-cli read ITEM_KEY --start-page 4 --end-page 4 --format image --out pages/
zotero-cli read ITEM_KEY --start-page 4 --format image --rect 0.2,0.3,0.35,0.2 --out zoom/   # zoom into one panel
```

Zoom into any panel before describing it: read the axis values, thresholds, which curve is which colour, and every
arrow and repression bar in a network diagram. Do not describe what you have not seen.

`rect` is `x,y,width,height`, normalised 0 to 1 to the page as displayed, from the top-left corner.

## 3. What to write

### Figure boxes: one per sub-panel, never one per figure

- Every lettered panel (1A, 1B, ...) gets its own box.
- Unlettered figures are split into their natural parts: a network diagram into its regions and legend, a grid of
  small plots into one box per plot, a text box into one box per paragraph.
- Do not bury the figure. For large regions of a dense diagram, box just the region's label and put the explanation in
  the comment.
- Tables: usually one box over the table, with the column guide and row notes in its comment.
- Comment format: start with `FIG 2B |` (or `TABLE 1 |`), then what is plotted (axes, units, colours, what one point
  is); what to notice, with the actual numbers; why the panel matters and which panel it pairs with; for diagrams,
  every edge; caveats such as weak statistics, small n, or text that disagrees with the figure.
- Use capitals sparingly, for the one phrase that is the point (`THE KEY CONTROL`).

### Body highlights

About 25 to 50 per paper: the central claim, the key design choices, each main finding, every limitation the authors
admit, and places where the text overreaches. Comments explain rather than paraphrase: why it matters, what it
connects to, what to question.

### Reading note

Add one note to the item: a first line `Author Year, Journal vol:pages. Title.`, then `QUESTION`, `APPROACH`,
`MAIN FINDINGS` (bullets), `DISCUSSION POINTS` (numbered) and an `ANNOTATION KEY` paragraph stating the colour scheme.

```bash
zotero-cli notes create --item-key ITEM_KEY --text - < note.md
```

### Writing style

Plain words, defined terms, no filler, no superlatives. Give numbers as the paper states them. Mention follow-up work
only when you are sure of the reference.

## 4. Plan, check, then write in one batch

Write the plan as a JSON Lines file, one annotation per line:

```json
{"page": 1, "rect": "0.10,0.07,0.80,0.145", "comment": "CONTEXT ...", "color": "orange"}
{"page": 1, "text": "exact words from the page", "comment": "PROBLEM. ...", "color": "yellow"}
```

1. Dry run, and read every `matched_text`. Fix misses and overshoots (shorten a phrase until it matches cleanly):
   ```bash
   zotero-cli --json annotations batch --attachment-key ATTACH_KEY --file plan.jsonl --dry-run
   ```
2. Check every box against the page images: each one covers exactly its panel, label included, and does not swallow
   its neighbour. On a rotated (landscape) page, open it in Zotero after writing and look.
3. Write for real (the same command without `--dry-run`) and confirm `failed: 0`. Then count what was written:
   ```bash
   zotero-cli --json annotations list --item-key ITEM_KEY --limit 500
   ```

## Gotchas

- Do not re-run a batch that succeeded: it creates duplicates. To redo, delete the old annotations permanently first
  (trashed annotations still show in Zotero's reader).
- `annotations list` fields are `type`, `color_category`, `page`, `text` and `comment`.
- Highlight text is searched on the given page and two either side, and tolerates hyphenation and curly quotes; copy
  it from `read`.
- Page numbers in a plan are PDF page indices starting at 1, not the printed journal page numbers.
