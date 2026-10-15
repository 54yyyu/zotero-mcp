"""scripts/site_release.py on a tiny copy of the website."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "site_release", Path(__file__).resolve().parent.parent / "scripts" / "site_release.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

HEADER = '<a class="version" href="x" title="Latest release">v0.1.0</a>'
FOOTER = "<span>Zotero MCP v0.1.0 &middot; MIT</span>"
INDEX = f"""{HEADER}
      <div class="releases">
        <article class="release lead">
          <div class="release-tag">0.1.0<small>Jan 2026</small></div>
        </article>
      </div>
<p>A skill costs <span data-cost="skill_frontmatter">90</span> tokens.</p>
<ul class="bars">
<li class="bar" data-cost="all"><small>50 tools, every request</small><span style="--w:100%"></span><span class="bar-val">1,000</span></li>
<li class="bar" data-cost="default"><small>40 tools, every request</small><span style="--w:80%"></span><span class="bar-val">800</span></li>
<li class="bar" data-cost="skill_body"><small>only when used</small><span style="--w:10%"></span><span class="bar-val">100</span></li>
<li class="bar" data-cost="skill_frontmatter"><small>always</small><span style="--w:1%"></span><span class="bar-val">90</span></li>
</ul>
{FOOTER}
"""
DOC = '{h}\n<span class="kicker">{k} &middot; v0.1.0</span>\n{f}\n'
REPORT = {
    "mcp": [
        {"profile": "(unset - default)", "tools": 41, "tokens": 16815},
        {"profile": "none", "tools": 34, "tokens": 14317},
        {"profile": "all", "tools": 53, "tokens": 20942},
    ],
    "skill": {"frontmatter": 98, "skill_total": 2400, "reference": 3758},
}
HIGHLIGHTS = """---
title: Faster {search}
date: Feb 2026
---
Search doesn't wait. Use `a<b` & see [#7](https://x/7).

> Rebuild once. With fixes from @someone (#7).
"""


@pytest.fixture
def site(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "agent/index.html").write_text(f"{HEADER}\n{FOOTER}\n")
    (tmp_path / "api.html").write_text(DOC.format(h=HEADER, k="Reference", f=FOOTER))
    (tmp_path / "installation.html").write_text(DOC.format(h=HEADER, k="Guide", f=FOOTER))
    costs = tmp_path / "costs.json"
    costs.write_text(__import__("json").dumps(REPORT))
    return tmp_path, costs


def run(site, costs, *extra):
    return sr.main(["--version", "0.2.0", "--site", str(site), "--costs", str(costs), *extra])


def test_bumps_every_version_string_and_rewrites_costs(site):
    root, costs = site
    assert run(root, costs) == 0
    for name in sr.PAGES:
        text = (root / name).read_text()
        assert "v0.1.0" not in text and "v0.2.0" in text
    index = (root / "index.html").read_text()
    assert "<small>53 tools" in index and "<small>41 tools" in index
    assert 'style="--w:100%"></span><span class="bar-val">20,942<' in index
    assert '--w:80.3%"></span><span class="bar-val">16,815<' in index
    assert '--w:11.5%"></span><span class="bar-val">2,400<' in index
    assert '--w:0.47%"></span><span class="bar-val">98<' in index
    assert '<span data-cost="skill_frontmatter">98</span>' in index
    # no highlights file: What's new is untouched
    assert index.count("<article") == 1 and "release lead" in index


def test_highlights_become_the_lead_release(site):
    root, costs = site
    hl = root / "hl.md"
    hl.write_text(HIGHLIGHTS)
    assert run(root, costs, "--highlights", str(hl)) == 0
    index = (root / "index.html").read_text()
    assert index.count("release lead") == 1
    assert index.index('release-tag">0.2.0<small>Feb 2026') < index.index('<article class="release">')
    assert '<h3>Faster <span class="hl c-yellow">search</span></h3>' in index
    assert ("<p>Search doesn&rsquo;t wait. Use <code>a&lt;b</code> &amp; see "
            '<a href="https://x/7">#7</a>.</p>') in index
    assert '<p class="fine">Rebuild once. With fixes from @someone (#7).</p>' in index
    # a rerun neither duplicates the entry nor fails
    assert run(root, costs, "--highlights", str(hl)) == 0
    assert (root / "index.html").read_text().count("0.2.0<small>") == 1


def test_missing_hook_fails_and_writes_nothing(site):
    root, costs = site
    api = root / "api.html"
    api.write_text(api.read_text().replace("Reference &middot; v0.1.0", "Reference"))
    assert run(root, costs) == 1
    assert "v0.1.0" in (root / "index.html").read_text()


def test_missing_cost_bar_fails(site):
    root, costs = site
    index = root / "index.html"
    index.write_text(index.read_text().replace(' data-cost="default"', ""))
    assert run(root, costs) == 1


def test_credits_from_merge_subjects():
    subjects = [
        "Merge #689: feat(toolsets): semantic search (#572) (#689, @falcon1-1-code)",
        "Merge pull request #684 from sghng: Voyage input_type",
        "Merge pull request #690 from 54yyyu/maint/0.14.1",
        "Merge fix/706-startup-update: run out of process (#706)",
        "Merge #693: fix(prompts): fall back (#693, @falcon1-1-code)",
        "Merge #694: fix(search): reject unknown fields (#694, @falcon1-1-code)",
        "Merge #695: fix(search): accept one condition (#695, @falcon1-1-code)",
    ]
    assert sr.credits_from_subjects(subjects) == (
        "With fixes from @falcon1-1-code (#689, #693 to #695) and @sghng (#684).")
    assert sr.credits_from_subjects(["Merge pull request #1 from 54yyyu/x"]) == ""
