# Skills for the Zotero chat panel

A skill is a folder with a `SKILL.md`: a short frontmatter (`name` and a one-line `description`), then plain Markdown
instructions your agent follows, using `zotero-cli` and its own shell. It may also hold reference files it points to
(`.md`, `.txt`, `.csv`, images). No scripts are needed.

```
annotate-paper/
  SKILL.md
```

The description decides when a skill is used, so make it say what the skill does and when ("Use when asked to annotate
a paper"). Keep the body to numbered steps, each with a way to check it worked.

## Use one

1. Copy the folder somewhere on your computer and read the whole `SKILL.md`. A skill tells your agent what to do, and the
   agent acts with your permissions: only add skills you trust or wrote.
2. In the panel, open Settings > **Skills and prompts** > **Add skill…** and pick its `SKILL.md`. The panel shows the
   full text and every file it will copy or leave out before anything is copied (scripts and links are left out).
3. Type `/annotate-paper` in the message box with a paper open, or pin it as a button on a new chat.

The panel never downloads skills. **Create with the agent** in the same card writes a new one with you.

## In this folder

- [`annotate-paper`](annotate-paper/SKILL.md): reads the open paper and writes study annotations into the PDF (a
  context box, a four-part summary, one box per figure panel, colour-coded highlights) plus a reading note. The colour
  scheme is a table at the top, there to be changed.
