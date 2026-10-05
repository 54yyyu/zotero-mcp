"""Note text to Zotero note-editor HTML: markdown, the shared tag subset, math, and the allow-list sanitizer."""

import re

import pytest

from zotero_mcp.note_html import append_note_html, sanitize_note_html, to_note_html


def body(html: str) -> str:
    return re.sub(r'^<div data-schema-version="\d">|</div>$', "", html).replace("\n", "")


def test_markdown_structure_becomes_editor_tags():
    html = body(to_note_html(
        "## Result\n\n**B**, *i*, ~~gone~~, `code`.\n\n- a\n- b\n\n1. one\n2. two\n\n> quoted\n\n---\n\n"
        "| A | B |\n|---|---|\n| 1 | 2 |\n\n```bash\necho \"$HOME\" < x\n```\n\nSee [OSF](https://osf.io/x)."
    ))
    assert "<h2>Result</h2>" in html
    assert "<p><strong>B</strong>, <em>i</em>, <s>gone</s>, <code>code</code>.</p>" in html
    assert "<ul><li>a</li><li>b</li></ul><ol><li>one</li><li>two</li></ol>" in html
    assert "<blockquote><p>quoted</p></blockquote><hr>" in html
    assert "<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>" in html
    assert '<pre><code>echo "$HOME" &lt; x</code></pre>' in html, "code is escaped and never math"
    assert '<a href="https://osf.io/x">OSF</a>' in html


def test_plain_text_keeps_its_lines():
    assert to_note_html("Line one\nline two\n\nLine three") == (
        '<div data-schema-version="8"><p>Line one<br>\nline two</p>\n<p>Line three</p></div>'
    )


def test_math_uses_the_editors_math_nodes():
    html = to_note_html(
        "Ratio $R = c_w / c_b$, money $5 and $10, \\(x\\) and \\[y^2\\] inline.\n\n"
        "$$\\Delta = \\ln\\frac{a}{b} < 1$$\n\n```math\na \\& b\n```"
    )
    assert html.startswith('<div data-schema-version="9">')
    assert '<span class="math">$R = c_w / c_b$</span>' in html
    assert "money $5 and $10" in html
    assert '<span class="math">$x$</span>' in html
    assert '<span class="math">$\\displaystyle y^2$</span>' in html
    assert '<pre class="math">$$\\Delta = \\ln\\frac{a}{b} &lt; 1$$</pre>' in html
    assert '<pre class="math">$$a \\&amp; b$$</pre>' in html


def test_the_shared_tag_subset_maps_to_the_editors_marks():
    html = body(to_note_html(
        'H<sub>2</sub>O, x<sup>2</sup>, <u>under</u>, <s>old</s>, <mark>key</mark>, '
        '<span style="color: red">red</span>, <span style="background-color: blue; color: #123456">both</span>, '
        '<span style="color: grey">grey</span>.'
    ))
    assert body(html) == (
        "<p>H<sub>2</sub>O, x<sup>2</sup>, <u>under</u>, <s>old</s>, "
        '<span style="background-color: #ffd40080">key</span>, <span style="color: #ff2020">red</span>, '
        '<span style="background-color: #2ea8e580; color: #123456">both</span>, '
        '<span style="color: #7e8386">grey</span>.</p>'
    )


@pytest.mark.parametrize("hostile", [
    "<script>alert(1)</script>after",
    '<img src=x onerror="alert(1)">after',
    '<a href="javascript:alert(1)">after</a>',
    '[x](javascript:alert(1)) after',
    '<span style="color: url(javascript:alert(1))">after</span>',
    '<span style="color: expression(alert(1)); position: fixed" onclick="alert(1)">after</span>',
    "<iframe src=//evil></iframe><style>p{display:none}</style>after",
    '<svg><g><path d="M0"/><script>alert(1)</script></svg>after',
    '<div onmouseover="alert(1)" style="background: url(x)"><p>after</p></div>',
    "<math><mi>x</mi></math>after",
])
def test_hostile_input_keeps_only_text(hostile):
    for html in (to_note_html(hostile), sanitize_note_html(hostile)):
        assert "after" in html
        tags = " ".join(re.findall(r"<[^>]*>", html)).lower()  # text is escaped, so only real tags can carry anything
        for bad in ("<script", "onerror", "onclick", "onmouseover", "javascript:", "url(", "expression", "<iframe",
                    "<style", "<svg", "<math", "position", "<img", "<a "):
            assert bad not in tags, (hostile, html)


def test_zoteros_own_markup_survives_a_round_trip():
    note = (
        '<div data-schema-version="9" data-citation-items="%5B%5D"><h1>T</h1>'
        '<p dir="rtl" data-indent="1" style="text-align: center; padding-left: 40px">c</p>'
        '<p><span class="citation" data-citation="%7B%22citationItems%22%3A%5B%5D%7D">(Bell, 2017)</span> '
        '<span class="highlight" data-annotation="%7B%7D">quoted</span></p>'
        '<p><img data-attachment-key="ABCD1234" width="400" height="300"></p>'
        '<p><u style="text-decoration-color: #ff2020">u</u> <span style="text-decoration: line-through">s</span></p>'
        '<pre class="math">$$x$$</pre><ol start="3"><li>three</li></ol>'
        '<table><tbody><tr><td colspan="2" colwidth="100,120">wide</td></tr></tbody></table></div>'
    )
    assert to_note_html(note) == note, "HTML input that is already the editor's vocabulary is kept as it is"
    # foreign attributes on it are not
    assert sanitize_note_html('<p style="color: red; text-align: left" onclick="x" class="y">a</p>') == '<p style="text-align: left">a</p>'
    assert sanitize_note_html('<img data-attachment-key="bad key" width="1e9">') == ""


def test_append_goes_inside_the_wrapper():
    assert append_note_html('<div data-schema-version="8"><p>old</p></div>', to_note_html("new $x$")) == (
        '<div data-schema-version="9"><p>old</p><p>new <span class="math">$x$</span></p></div>'
    )
    assert append_note_html("<p>old</p>", to_note_html("<p>more</p>")) == "<p>old</p><p>more</p>"
    assert append_note_html("", to_note_html("hi")) == '<div data-schema-version="8"><p>hi</p></div>'
    assert append_note_html("<h1>T</h1>", to_note_html("hi"), at_start=True) == '<div data-schema-version="8"><h1>T</h1><p>hi</p></div>'
