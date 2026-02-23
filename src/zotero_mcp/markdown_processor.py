"""
Markdown-aware chunking utilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MarkdownChunk:
    text: str
    char_start: int
    char_end: int
    section_path: str
    page_start: int | None = None
    page_end: int | None = None


def _split_sections(markdown_text: str) -> list[tuple[str, str, int, int]]:
    """
    Return tuples of (section_path, section_text, char_start, char_end).
    """
    lines = markdown_text.splitlines(keepends=True)
    heading_stack: list[str] = []
    sections: list[tuple[str, str, int, int]] = []

    current_text: list[str] = []
    current_start = 0
    cursor = 0
    current_path = "Document"

    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)\s*$")

    for line in lines:
        m = heading_pattern.match(line.strip("\n"))
        if m:
            if current_text:
                joined = "".join(current_text).strip()
                if joined:
                    sections.append((current_path, joined, current_start, cursor))
            level = len(m.group(1))
            title = m.group(2).strip()
            if len(heading_stack) >= level:
                heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_path = " > ".join(heading_stack) if heading_stack else "Document"
            current_text = []
            current_start = cursor
        current_text.append(line)
        cursor += len(line)

    if current_text:
        joined = "".join(current_text).strip()
        if joined:
            sections.append((current_path, joined, current_start, cursor))

    if not sections:
        text = markdown_text.strip()
        if text:
            sections.append(("Document", text, 0, len(markdown_text)))
    return sections


def _token_windows(text: str, window: int, overlap: int) -> list[tuple[int, int, str]]:
    # Approximate tokenization with whitespace units for robustness.
    words = re.findall(r"\S+\s*", text)
    if not words:
        return []
    out: list[tuple[int, int, str]] = []
    i = 0
    step = max(1, window - overlap)
    while i < len(words):
        segment_words = words[i : i + window]
        if not segment_words:
            break
        segment = "".join(segment_words).strip()
        if segment:
            char_s = len("".join(words[:i]))
            char_e = char_s + len("".join(segment_words))
            out.append((char_s, char_e, segment))
        i += step
    return out


def chunk_markdown(
    markdown_text: str,
    window_tokens: int = 600,
    overlap_tokens: int = 100,
) -> list[MarkdownChunk]:
    chunks: list[MarkdownChunk] = []
    for section_path, section_text, section_char_start, _ in _split_sections(markdown_text):
        windows = _token_windows(section_text, window_tokens, overlap_tokens)
        for rel_start, rel_end, text in windows:
            chunks.append(
                MarkdownChunk(
                    text=text,
                    char_start=section_char_start + rel_start,
                    char_end=section_char_start + rel_end,
                    section_path=section_path,
                )
            )
    return chunks

