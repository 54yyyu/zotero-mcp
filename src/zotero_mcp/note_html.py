"""Note text to the HTML Zotero's note editor stores.

Markdown (with $math$ and the small tag subset the chat panel shares: <u> <s> <sub> <sup> <mark> and
<span style="color: …; background-color: …">) becomes note-editor HTML; HTML input is kept to the editor's own
vocabulary. Everything passes one allow-list sanitizer, so no script, handler, foreign style or unknown tag is
stored. Zotero's own attributes (citations, annotation highlights, embedded images, alignment, indent) survive, so
a note read with --raw-html, edited and written back keeps them.
"""

from __future__ import annotations

import re
from html import escape
from html.parser import HTMLParser

from markdown_it import MarkdownIt

#: The note editor's palettes: text colours, and highlight colours (its own at 50%).
TEXT_COLORS = {
    "red": "#ff2020", "orange": "#ff7700", "yellow": "#ffcb00", "green": "#4eb31c",
    "purple": "#7953e3", "magenta": "#eb52f7", "blue": "#05a2ef", "gray": "#7e8386",
}
HIGHLIGHT_COLORS = {
    "red": "#ff666680", "orange": "#f1983780", "yellow": "#ffd40080", "green": "#5fb23680",
    "purple": "#a28ae580", "magenta": "#e56eee80", "blue": "#2ea8e580", "gray": "#aaaaaa80",
}
_COLOR = re.compile(r"#(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})|rgba?\(\s*[\d.]+%?\s*(?:,\s*[\d.]+%?\s*){2,3}\)", re.I)

_TAGS = {
    "div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "code", "strong", "b",
    "em", "i", "u", "s", "strike", "del", "sub", "sup", "span", "a", "br", "hr", "table", "thead", "tbody", "tr",
    "th", "td", "img", "mark",
}
_VOID = {"br", "hr", "img"}
#: Gone with everything inside them.
_DROP = {"script", "style", "iframe", "object", "embed", "noscript", "template", "svg", "math", "head", "title",
         "textarea", "select", "button", "form", "frame", "frameset", "canvas", "video", "audio"}
_BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote", "div"}
_DIGITS = re.compile(r"\d{1,6}")
_ATTRS = {  # tag -> attribute -> what its value must match (fullmatch)
    "div": {"data-schema-version": _DIGITS, "data-citation-items": re.compile(r"[^\x00]*")},
    "span": {"class": re.compile(r"math|citation|highlight|underline"), "data-citation": re.compile(r"[^\x00]*"),
             "data-annotation": re.compile(r"[^\x00]*")},
    "pre": {"class": re.compile(r"math")},
    "a": {"href": re.compile(r"(?:https?|zotero|mailto):[^\s\x00]*", re.I), "title": re.compile(r"[^\x00]*")},
    "img": {"data-attachment-key": re.compile(r"[A-Z0-9]{8}"), "width": _DIGITS, "height": _DIGITS,
            "alt": re.compile(r"[^\x00]*"), "data-annotation": re.compile(r"[^\x00]*"),
            "src": re.compile(r"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=\s]+")},
    "ol": {"start": _DIGITS},
    "td": {"colspan": _DIGITS, "rowspan": _DIGITS, "colwidth": re.compile(r"[\d,]{1,40}")},
    "th": {"colspan": _DIGITS, "rowspan": _DIGITS, "colwidth": re.compile(r"[\d,]{1,40}")},
}
_BLOCK_ATTRS = {"dir": re.compile(r"ltr|rtl|auto"), "data-indent": _DIGITS}
_STYLE = {  # property -> allowed value (None: a colour)
    "color": None, "background-color": None, "text-decoration-color": None,
    "text-align": re.compile(r"left|right|center|justify|start|end"), "padding-left": re.compile(r"\d{1,4}px"),
    "text-decoration": re.compile(r"underline|line-through"), "text-decoration-line": re.compile(r"underline|line-through"),
    "vertical-align": re.compile(r"sub|super"),
}


def _color(value: str, palette: dict) -> str | None:
    v = value.strip().lower()
    v = "gray" if v == "grey" else v
    return palette.get(v) or (v if _COLOR.fullmatch(v) else None)


def _style(tag: str, css: str) -> str:
    """Only the declarations the editor itself writes, with values checked; colour names become its palette."""
    out = []
    for decl in css.split(";"):
        prop, _, value = decl.partition(":")
        prop, value = prop.strip().lower(), value.strip()
        if prop == "background":
            prop = "background-color"
        if prop not in _STYLE or not value:
            continue
        rule = _STYLE[prop]
        if rule is None:
            value = _color(value, HIGHLIGHT_COLORS if prop == "background-color" else TEXT_COLORS)
        elif not rule.fullmatch(value.lower()):
            value = None
        if value and (prop in ("text-align", "padding-left")) == (tag in _BLOCK):
            out.append(f"{prop}: {value}")
    return "; ".join(out)


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[str] = []
        self.drop_tag = ""  # inside <script>, <svg>, …: everything up to its own closing tag goes
        self.dropping = 0

    def handle_starttag(self, tag, attrs):
        if self.dropping:
            self.dropping += tag == self.drop_tag
            return
        if tag in _DROP:
            self.drop_tag, self.dropping = tag, 1
            return
        if tag not in _TAGS:
            return  # unknown: its content stays, the tag goes
        allowed = {**_ATTRS.get(tag, {}), **(_BLOCK_ATTRS if tag in _BLOCK else {})}
        kept = []
        style = ""
        for name, value in attrs:
            value = value or ""
            if name == "style":
                style = _style(tag, value)
            elif name in allowed and allowed[name].fullmatch(value):
                kept.append(f' {name}="{escape(value)}"')
        if tag == "mark":  # the editor has no <mark>: its highlight mark is a background colour
            tag, style = "span", f"background-color: {HIGHLIGHT_COLORS['yellow']}"
        if style and tag in ("span", "u", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            kept.append(f' style="{escape(style)}"')
        if tag == "img" and not any(k.startswith((" data-attachment-key=", " src=")) for k in kept):
            return  # an image that points at nothing we allow
        if tag == "span" and not kept:
            self.stack.append("")  # a bare span is noise; its content stays
            return
        self.out.append(f"<{tag}{''.join(kept)}>")
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if self.dropping or tag in _DROP:
            return
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.dropping:
            self.dropping -= tag == self.drop_tag
            return
        tag = "span" if tag == "mark" else tag
        if tag not in _TAGS or tag in _VOID:
            return
        # close up to the matching open tag (an unbalanced close is ignored)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i] == tag or (tag == "span" and self.stack[i] == ""):
                for t in reversed(self.stack[i:]):
                    if t:
                        self.out.append(f"</{t}>")
                del self.stack[i:]
                return

    def handle_data(self, data):
        if not self.dropping:
            self.out.append(escape(data, quote=False))

    def result(self) -> str:
        self.close()
        return "".join(self.out) + "".join(f"</{t}>" for t in reversed(self.stack) if t)


def sanitize_note_html(html: str) -> str:
    """HTML kept to the note editor's vocabulary (see the module docstring)."""
    s = _Sanitizer()
    s.feed(html)
    return s.result()


# ── markdown ──────────────────────────────────────────────────────────────

_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})[^\n]*\n.*?(?:^ {0,3}\2[ \t]*$|\Z)", re.M | re.S)
_CODESPAN = re.compile(r"(`+)(?!`).+?(?<!`)\1", re.S)
_MATH = re.compile(
    r"\$\$(?P<d>(?:\\.|[^\\$])+?)\$\$"
    r"|\\\[(?P<b>.+?)\\\]"
    r"|\\\((?P<p>.+?)\\\)"
    r"|\$(?=[^\s$])(?P<i>(?:\\.|[^\\$\n]|\n(?![ \t]*\n))+?)(?<=[^\s\\])\$(?!\d)",
    re.S,
)
_PH = "ZQMATH{}Q"


def _protect_math(text: str) -> tuple[str, list[tuple[str, bool]]]:
    """Formulas out of the way of markdown (as placeholders), never inside code."""
    maths: list[tuple[str, bool]] = []

    def hold(tex: str, display: bool) -> str:
        maths.append((tex.strip(), display))
        return _PH.format(len(maths) - 1)

    def outside_code(chunk: str) -> str:
        parts, last = [], 0
        for m in _CODESPAN.finditer(chunk):
            parts.append(_MATH.sub(lambda x: hold(next(v for v in x.groups() if v is not None), x.group("d") is not None or x.group("b") is not None), chunk[last:m.start()]))
            parts.append(m.group(0))
            last = m.end()
        parts.append(_MATH.sub(lambda x: hold(next(v for v in x.groups() if v is not None), x.group("d") is not None or x.group("b") is not None), chunk[last:]))
        return "".join(parts)

    out, last = [], 0
    for m in _FENCE.finditer(text):
        out.append(outside_code(text[last:m.start()]))
        block = m.group(0)
        lang = block.split("\n", 1)[0].strip(" `~").lower()
        if lang == "math":  # a ```math fence is a display formula
            body = block.split("\n", 1)[1] if "\n" in block else ""
            body = re.sub(r"\n? {0,3}(`{3,}|~{3,})[ \t]*$", "", body)
            out.append("\n\n" + hold(body, True) + "\n\n")
        else:
            out.append(block)
        last = m.end()
    out.append(outside_code(text[last:]))
    return "".join(out), maths


_MD = MarkdownIt("commonmark", {"html": True, "breaks": True}).enable(["table", "strikethrough"])


def markdown_to_note_html(text: str) -> str:
    """Markdown with $math$ and the tag subset, as note-editor HTML in its schema wrapper."""
    held, maths = _protect_math(text.replace("\r\n", "\n"))
    html = _MD.render(held)

    def math(i: int, block: bool) -> str:
        tex, display = maths[i]
        if display and block:
            return f'<pre class="math">$${escape(tex, quote=False)}$$</pre>'
        inline = r"\displaystyle " + tex if display else tex  # display math inside a sentence stays inline
        return f'<span class="math">${escape(inline, quote=False)}$</span>'

    html = re.sub(r"<p>ZQMATH(\d+)Q</p>", lambda m: math(int(m.group(1)), True), html)
    html = re.sub(r"ZQMATH(\d+)Q", lambda m: math(int(m.group(1)), False), html)
    version = 9 if maths else 8
    return f'<div data-schema-version="{version}">{sanitize_note_html(html).strip()}</div>'


_HTML_START = re.compile(r"\s*<(?:div|p|h[1-6]|ul|ol|table|pre|blockquote)[\s>]", re.I)


def to_note_html(text: str) -> str:
    """Note text as the editor's HTML: HTML input (it starts with a block tag) is sanitized, anything else is markdown."""
    return sanitize_note_html(text) if _HTML_START.match(text) else markdown_to_note_html(text)


_WRAPPER = re.compile(r'^\s*<div data-schema-version="(\d+)"([^>]*)>(.*)</div>\s*$', re.S)


def append_note_html(existing: str, addition: str, at_start: bool = False) -> str:
    """`addition` added at the end of a note, inside its schema wrapper when it has one. `at_start`: `existing` is a
    heading to put first inside `addition`'s wrapper."""
    if at_start:
        new = _WRAPPER.match(addition)
        return f'<div data-schema-version="{new.group(1)}"{new.group(2)}>{existing}{new.group(3)}</div>' if new else existing + addition
    old, new = _WRAPPER.match(existing or ""), _WRAPPER.match(addition)
    if not old:
        return (existing or "") + (new.group(3) if new and existing else addition)
    version = max(int(old.group(1)), int(new.group(1)) if new else 0)
    return f'<div data-schema-version="{version}"{old.group(2)}>{old.group(3)}{new.group(3) if new else addition}</div>'
