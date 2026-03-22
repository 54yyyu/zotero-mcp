# Zotero MCP Feature Test Plan

Use this document as instructions for Claude Co-Work. Copy/paste each section as a prompt, or give the whole document at once and ask Claude to work through it step by step.

**Prerequisites:**
- Zotero 8 is running on this computer
- The local API is enabled in Zotero preferences ("Allow other applications on this computer to communicate with Zotero")
- The modified zotero-mcp is installed (run `zotero-mcp version` in terminal to verify)

**Important:** After each write operation, open Zotero and verify the change appeared. This confirms hybrid mode is working (local reads + web writes syncing back to local).

---

## Phase 1: Read-Only Features (Safe — No Changes to Library)

### Test 1.1: Search Collections
```
Search my Zotero collections for any collection that exists.
List all collections you can find.
```
**Verify:** The results show collection names and keys that match what you see in Zotero's left sidebar.

### Test 1.2: Find Duplicates
```
Search my library for duplicate items using the "both" method
(check both titles and DOIs). Limit to 50 results.
```
**Verify:** If duplicates exist, they should be grouped and show titles, keys, and DOIs for comparison. If no duplicates exist, it should say so clearly.

### Test 1.3: Find Duplicates (Title Only)
```
Search for duplicates using only title matching.
```
**Verify:** May catch more matches than "both" since it ignores DOI differences.

### Test 1.4: PDF Outline Extraction
```
Pick any item in my library that has a PDF attachment and extract
its table of contents / outline.
```
**Verify:** Should show an indented list of sections with page numbers, or a clear message if the PDF has no outline.

### Test 1.5: Semantic Search (if configured)
```
Do a semantic search for "machine learning" in my library.
```
**Verify:** Returns relevant results ranked by similarity. (Skip if semantic search isn't set up.)

---

## Phase 2: Collection Management (Creates New Collections — Easy to Clean Up)

### Test 2.1: Create a Collection
```
Create a new collection in my Zotero library called "MCP Test Collection".
```
**Verify:** Open Zotero, look in the left sidebar — "MCP Test Collection" should appear. Note the collection key from the response.

### Test 2.2: Create a Subcollection
```
Create a subcollection called "Test Subcollection" inside
"MCP Test Collection".
```
**Verify:** In Zotero, expand "MCP Test Collection" — "Test Subcollection" should appear nested underneath.

### Test 2.3: Search for the New Collection
```
Search for collections matching "MCP Test".
```
**Verify:** Both "MCP Test Collection" and "Test Subcollection" should appear with their keys.

---

## Phase 3: Adding Items (Creates New Items — Can Be Deleted After)

### Test 3.1: Add by DOI
```
Add this paper to my Zotero library by DOI: 10.1038/nature12373

This is a well-known Nature paper. After adding, tell me the item key
and what metadata was pulled from CrossRef.
```
**Verify:** Open Zotero — a new item should appear with full metadata (title, authors, journal, date, DOI). Check that the title, authors, and journal name look correct.

### Test 3.2: Add by DOI with Tags and Collection
```
Add this paper by DOI: 10.1126/science.1157996
Add it to the "MCP Test Collection" and tag it with "test" and "science".
```
**Verify:** Item appears in Zotero with correct metadata, is inside "MCP Test Collection", and has both tags.

### Test 3.3: Add by arXiv URL
```
Add this arXiv paper to my library: https://arxiv.org/abs/1706.03762
Tag it with "transformers" and "deep learning".
```
**Verify:** Item appears as a preprint with title "Attention Is All You Need" (or similar), correct authors, and both tags.

### Test 3.4: Add by DOI URL
```
Add this paper: https://doi.org/10.1145/3065386
```
**Verify:** Should recognize this as a DOI URL, resolve it via CrossRef, and create the item with full metadata.

### Test 3.5: Add by Generic URL
```
Add this webpage to my library: https://www.zotero.org/support/quick_start_guide
```
**Verify:** Creates a "webpage" type item with the URL set.

---

## Phase 4: Updating Items

### Test 4.1: Update Title
```
Find the item you just added by DOI 10.1038/nature12373 and update
its title to add "[TEST]" at the beginning.
```
**Verify:** In Zotero, the item's title now starts with "[TEST]".

### Test 4.2: Add Tags Incrementally
```
Add the tags "neuroscience" and "updated-by-mcp" to that same item,
without removing any existing tags.
```
**Verify:** Item now has its original tags PLUS the two new ones.

### Test 4.3: Remove a Tag
```
Remove the tag "updated-by-mcp" from that item.
```
**Verify:** Tag is gone, but all other tags remain.

### Test 4.4: Update Abstract
```
Update the abstract of that item to: "This is a test abstract
set by the MCP test plan."
```
**Verify:** Abstract field in Zotero shows the new text.

### Test 4.5: Restore Title
```
Remove the "[TEST]" prefix from that item's title, restoring the
original title.
```
**Verify:** Title is back to normal.

---

## Phase 5: Collection Membership Management

### Test 5.1: Add Items to Collection
```
Take the items you added in Phase 3 (the Nature paper and the arXiv
paper) and add them both to "MCP Test Collection" if they aren't
already there.
```
**Verify:** Both items appear under "MCP Test Collection" in Zotero.

### Test 5.2: Remove from Collection
```
Remove the arXiv paper from "MCP Test Collection".
```
**Verify:** The arXiv paper is no longer in "MCP Test Collection" but still exists in the main library (it wasn't deleted, just removed from the collection).

---

## Phase 6: Merge Duplicates (Dry-Run First)

### Test 6.1: Create Test Duplicates
```
Add this DOI twice to create a deliberate duplicate:
First: Add DOI 10.1016/j.cell.2015.11.015 with tag "copy-1"
Then: Add the same DOI 10.1016/j.cell.2015.11.015 again with tag "copy-2"
```
**Verify:** Two separate items with the same title/DOI appear in Zotero, each with different tags.

### Test 6.2: Find the Duplicates
```
Find duplicates in my library. The two items you just created should
appear as a duplicate group.
```
**Verify:** They appear grouped together with both keys shown.

### Test 6.3: Dry-Run Merge
```
Merge those duplicates. Use the first one (copy-1) as the keeper.
Do NOT confirm yet — just show me the dry-run preview of what would happen.
```
**Verify:** Response shows:
- Which item will be kept
- Which will be trashed
- What tags will be consolidated (both "copy-1" and "copy-2" should be listed)
- What children (notes, attachments) would be moved
- A message saying "Call again with confirm=True to execute"

### Test 6.4: Execute Merge
```
Go ahead and confirm the merge.
```
**Verify:**
- In Zotero, only one item remains with BOTH tags ("copy-1" and "copy-2")
- The duplicate should be in Zotero's Trash (check View > Show Trash)
- Any notes or attachments from the duplicate should now be under the keeper

---

## Phase 7: Add from File (Only If You Have a Local PDF)

### Test 7.1: Add a PDF
```
Add this PDF file to my Zotero library: [REPLACE WITH AN ACTUAL PDF PATH]

For example: /Users/eugenehawkin/Documents/some-paper.pdf
```
**Verify:** Item created in Zotero. If the PDF contained a DOI, metadata should be auto-populated. The PDF should be attached to the item.

*Skip this test if you don't have a convenient PDF file to test with.*

---

## Phase 8: Hybrid Mode Verification

### Test 8.1: Batch Update Tags by Tag Filter (Original Bug Fix)
```
Use zotero_batch_update_tags to find items with the tag "test"
(use the tag parameter, not just the query) and add the tag
"mcp-test-verified" to those items.
```
**Verify:** Items that had the "test" tag now also have "mcp-test-verified". The `tag` parameter filters by actual tag name, not just text search.

### Test 8.2: Batch Update Tags by Text Query
```
Use zotero_batch_update_tags with query "Attention" (text search)
and add the tag "attention-paper".
```
**Verify:** The arXiv "Attention Is All You Need" paper gets the new tag.

---

## Phase 9: Regression Tests (Bugs Fixed in Round 2)

These tests specifically verify bugs found during the first test run.

### Test 9.1: manage_collections Tool Works
```
Use zotero_manage_collections (NOT update_item) to add the Nature
paper from Phase 3 to "MCP Test Collection". Use the add_to parameter.
```
**Verify:** The item appears in the collection. This tests the fix for the "list indices must be integers" error.

### Test 9.2: update_item Preserves Existing Collections
```
First, check what collections the Nature paper is currently in.
Then use zotero_update_item with collections=["<MCP Test Collection key>"]
to add it to that collection.
Verify it's still in any collections it was in before.
```
**Verify:** The item is now in "MCP Test Collection" AND still in any collection it was already in. Collections should be merged, not replaced.

### Test 9.3: create_note with JSON String Tags
```
Create a note on the Nature paper with:
- title: "Test Note"
- text: "This is a test note created by the MCP test plan."
- tags: Pass the tags as a JSON string: '["test-note", "regression-test"]'
```
**Verify:** The note is created successfully (no Pydantic validation error), has both tags, and the returned key is a real item key (like "XGW2GIMC"), not "0".

### Test 9.4: create_note Returns Correct Key
```
Create another note on the Nature paper. Check that the returned
note key actually corresponds to the note (search for it by key).
```
**Verify:** The key returned in the success message matches the actual note. Previously this returned "0" instead of the real key.

### Test 9.5: Add by DOI with PDF Attachment
```
Add this open-access paper by DOI: 10.1371/journal.pone.0001636
Check if a PDF was automatically attached.
```
**Verify:** The item is created AND an open-access PDF is attached (check for a PDF icon next to the item in Zotero). If no PDF attached, the message should say "no open-access PDF found" (not an error).

### Test 9.6: Add arXiv Paper with PDF
```
Add this arXiv paper: https://arxiv.org/abs/2301.00774
Check if the arXiv PDF was automatically attached.
```
**Verify:** The preprint item is created AND the arXiv PDF is attached.

### Test 9.7: Merge Duplicates Actually Trashes
```
Create two duplicates (add DOI 10.1038/s41586-020-2649-2 twice with
tags "merge-test-1" and "merge-test-2").
Find them, then merge with confirm=True.
Check Zotero's Trash (View > Show Trash).
```
**Verify:** The duplicate appears in Zotero's Trash (not permanently deleted). The keeper has both tags. The trashed item can be restored if needed.

---

## Phase 10: PDF Cascade Sources

These tests verify that the 4-source PDF auto-attachment cascade works correctly.

### Test 10.1: PDF via Unpaywall (Gold OA)
```
Add this open-access paper by DOI: 10.1371/journal.pone.0185809
Check if a PDF was automatically attached and which source provided it.
```
**Verify:** PDF is attached. The response should mention Unpaywall as the source. This is a PLOS ONE paper (gold open access), so Unpaywall should always find it.

### Test 10.2: PDF via arXiv from CrossRef Metadata
```
Add this paper by DOI: 10.1103/PhysRevD.110.L081901
Check if a PDF was automatically attached.
```
**Verify:** PDF is attached. This is a Physical Review D paper that has an arXiv preprint. The CrossRef metadata contains a `has-preprint` relation pointing to the arXiv version. The cascade should find the arXiv PDF even though the journal version is paywalled.

### Test 10.3: PDF via PubMed Central
```
Add this paper by DOI: 10.1261/rna.053959.115
Check if a PDF was automatically attached.
```
**Verify:** PDF is attached. This paper is in PubMed Central (PMC). The cascade should find it via the NCBI ID converter API.

### Test 10.4: Graceful Failure (Paywalled, No OA)
```
Add this paper by DOI: 10.1016/j.tetlet.2019.151042
Check the response message about PDF attachment.
```
**Verify:** Item is created with full metadata, but NO PDF is attached. The response should include a clear message like "No open-access PDF found" — not an error or crash.

---

## Phase 11: attach_mode Parameter

### Test 11.1: Linked URL Mode
```
Add this paper by DOI: 10.1038/nature12373
Use attach_mode="linked_url" so it saves the URL without downloading.
```
**Verify:** Item is created. Instead of a downloaded PDF, there should be a linked URL attachment. In Zotero, the attachment icon will look different from a regular PDF — it will show as a link rather than a file.

### Test 11.2: Auto Mode Fallback
```
Add a paywalled paper by DOI: 10.1016/j.cell.2015.11.015
Use attach_mode="auto" (or don't specify, since auto is the default).
```
**Verify:** The cascade tries to download a PDF (fails because it's paywalled), then falls back to creating a linked URL attachment. The response should indicate the fallback happened.

---

## Phase 12: BetterBibTeX Citation Key Lookup

*Skip this phase if you don't have the BetterBibTeX plugin installed in Zotero.*

### Test 12.1: Look Up by Citation Key
```
Look up the paper with citation key "Smith2024" in my library.
(Replace "Smith2024" with an actual citation key from your library.)
```
**Verify:** Returns the correct paper with full metadata. If BetterBibTeX is installed and running, it should use the BBT API directly. If not, it falls back to searching the Extra field.

### Test 12.2: Citation Key Not Found
```
Look up the paper with citation key "NonexistentKey9999".
```
**Verify:** Returns a clear "not found" message, not an error or crash.

---

## Phase 13: Showcase Prompts

These test complex, multi-step requests that exercise several tools together.

### Test 13.1: Research Topic Collection
```
What are the three most seminal papers on Predictive Coding?
Can you please locate them for me and then create a predictive
coding collection in my Zotero library and add those three papers?
```
**Verify:** Claude identifies ~3 papers, adds them by DOI (with metadata and PDFs where available), creates a "Predictive Coding" collection, and adds all three papers to it. Check Zotero to confirm everything is there.

### Test 13.2: Annotation and Highlighting
```
Take a look at the 2025 paper on digital mindfulness interventions
by Wang et al. and highlight in green any sentences in the abstract,
discussion, or conclusion that you feel represent the core findings.
```
**Verify:** Claude finds the paper, reads its content, identifies key findings, and creates green highlight annotations on the relevant passages. Check the PDF in Zotero's reader to see the highlights.

*Note: This test requires the paper to be in your library with a PDF attached.*

---

## Cleanup

After all tests pass, clean up the test data:

```
Please help me clean up the test items:
1. List all items tagged "mcp-test-verified", "copy-1", "copy-2",
   "test", or "science" that were created today
2. I'll delete them manually from Zotero

Also, I'll manually delete the "MCP Test Collection" from Zotero's
left sidebar.
```

*Note: We don't have a delete-item tool (by design — deletion is too destructive for an MCP tool). Clean up manually in the Zotero app.*

---

## Results Summary

After running all tests, fill in this checklist:

| Test | Feature | Result |
|------|---------|--------|
| 1.1 | Search Collections | Pass / Fail |
| 1.2 | Find Duplicates (both) | Pass / Fail |
| 1.3 | Find Duplicates (title) | Pass / Fail |
| 1.4 | PDF Outline | Pass / Fail |
| 1.5 | Semantic Search | Pass / Fail / Skipped |
| 2.1 | Create Collection | Pass / Fail |
| 2.2 | Create Subcollection | Pass / Fail |
| 2.3 | Search New Collection | Pass / Fail |
| 3.1 | Add by DOI | Pass / Fail |
| 3.2 | Add by DOI + Tags + Collection | Pass / Fail |
| 3.3 | Add by arXiv URL | Pass / Fail |
| 3.4 | Add by DOI URL | Pass / Fail |
| 3.5 | Add by Generic URL | Pass / Fail |
| 4.1 | Update Title | Pass / Fail |
| 4.2 | Add Tags | Pass / Fail |
| 4.3 | Remove Tag | Pass / Fail |
| 4.4 | Update Abstract | Pass / Fail |
| 4.5 | Restore Title | Pass / Fail |
| 5.1 | Add to Collection | Pass / Fail |
| 5.2 | Remove from Collection | Pass / Fail |
| 6.1 | Create Test Duplicates | Pass / Fail |
| 6.2 | Find Duplicates | Pass / Fail |
| 6.3 | Dry-Run Merge | Pass / Fail |
| 6.4 | Execute Merge | Pass / Fail |
| 7.1 | Add from File | Pass / Fail / Skipped |
| 8.1 | Batch Update Tags (by tag) | Pass / Fail |
| 8.2 | Batch Update Tags (by query) | Pass / Fail |
| 9.1 | manage_collections Works | Pass / Fail |
| 9.2 | update_item Preserves Collections | Pass / Fail |
| 9.3 | create_note JSON String Tags | Pass / Fail |
| 9.4 | create_note Returns Correct Key | Pass / Fail |
| 9.5 | Add by DOI with PDF | Pass / Fail |
| 9.6 | Add arXiv with PDF | Pass / Fail |
| 9.7 | Merge Trashes (Not Deletes) | Pass / Fail |
| 10.1 | PDF via Unpaywall | Pass / Fail |
| 10.2 | PDF via arXiv from CrossRef | Pass / Fail |
| 10.3 | PDF via PubMed Central | Pass / Fail |
| 10.4 | Graceful Failure (No OA) | Pass / Fail |
| 11.1 | attach_mode linked_url | Pass / Fail |
| 11.2 | attach_mode auto fallback | Pass / Fail |
| 12.1 | BetterBibTeX Lookup | Pass / Fail / Skipped |
| 12.2 | Citation Key Not Found | Pass / Fail / Skipped |
| 13.1 | Showcase: Research Collection | Pass / Fail |
| 13.2 | Showcase: Annotation Highlighting | Pass / Fail |

**Total: 45 tests**
