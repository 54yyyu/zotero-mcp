#!/usr/bin/env python3
"""Update the project website (the gh-pages branch) for a release.

    python scripts/site_release.py --version 0.15.1 --site ../gh-pages \
        [--highlights site/releases/0.15.1.md] [--costs report.json]

Always: bumps every version string and rewrites the context-cost numbers
(from ``measure_context_cost.py --json``, or ``--costs``). With a highlights
file it also adds the release to "What's new" as the lead entry. Nothing is
written unless every expected pattern was found, so a site redesign that
moves a hook fails here instead of shipping a half-updated page.

Highlights file (``site/releases/X.Y.Z.md``)::

    ---
    title: Windows fixes, and full PDFs from {repositories}
    date: Oct 2026
    ---
    A paragraph. `code`, [links](https://...) and straight quotes work.

    Another paragraph.

    > Fine print. Without an @mention here, a "With fixes from ..." line
    > is added from the merge commits since the previous tag.

``{word}`` in the title gets the yellow highlight; ``date`` defaults to this month.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGES = ("index.html", "api.html", "installation.html", "agent/index.html")
VER = r"\d+\.\d+\.\d+"
# (pages, pattern, label): each page must match at least once.
VERSION_HOOKS = (
    (PAGES, rf'(class="version"[^>]*>v){VER}', "header badge"),
    (PAGES, rf"(Zotero MCP v){VER}(?= &middot;)", "footer"),
    (("api.html", "installation.html"), rf"((?:Reference|Guide) &middot; v){VER}", "kicker"),
)
LEAD = '<article class="release lead">'
RELEASES = '<div class="releases">\n'
MAINTAINER = "54yyyu"  # own PRs are not credited


class SiteError(Exception):
    pass


def bump_versions(pages: dict[str, str], version: str) -> None:
    for names, pattern, label in VERSION_HOOKS:
        for name in names:
            pages[name], n = re.subn(pattern, rf"\g<1>{version}", pages[name])
            if not n:
                raise SiteError(f"{name}: no {label} version string ({pattern})")


def inline(text: str) -> str:
    """Markdown-ish inline text to the site's HTML: `code`, [a](url), quotes."""
    out = []
    for i, part in enumerate(text.split("`")):
        if i % 2:
            out.append(f"<code>{html.escape(part, quote=False)}</code>")
            continue
        part = re.sub(r"&(?!#?\w+;)", "&amp;", part).replace("<", "&lt;").replace(">", "&gt;")
        part = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', part)
        part = part.replace("'", "&rsquo;")
        out.append(part)
    return "".join(out)


def parse_highlights(text: str) -> dict:
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not m:
        raise SiteError("highlights: missing --- frontmatter ---")
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    if not meta.get("title"):
        raise SiteError("highlights: frontmatter needs a title")
    paras, fine = [], []
    for block in re.split(r"\n\s*\n", m.group(2).strip()):
        lines = [ln.strip() for ln in block.splitlines()]
        if all(ln.startswith(">") for ln in lines):
            fine.append(" ".join(ln.lstrip("> ").strip() for ln in lines))
        else:
            paras.append(" ".join(lines))
    return {"title": meta["title"], "date": meta.get("date"), "paras": paras, "fine": " ".join(fine)}


def _numbers(nums: list[int]) -> str:
    nums, runs = sorted(set(nums)), []
    for n in nums:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    parts = []
    for a, b in runs:
        parts += [f"#{a} to #{b}"] if b - a >= 2 else [f"#{x}" for x in range(a, b + 1)]
    return ", ".join(parts)


def credits_from_subjects(subjects: list[str]) -> str:
    """'With fixes from @a (#1, #2) and @b (#3).' from merge subjects, oldest first."""
    by_user: dict[str, list[int]] = {}
    for s in subjects:  # "Merge #N: ... (#N, @user)" or "Merge pull request #N from user..."
        m = re.search(r"\(#(\d+), @([\w-]+)\)\s*$", s) or re.match(r"Merge pull request #(\d+) from ([\w-]+)", s)
        if m and m.group(2) != MAINTAINER:
            by_user.setdefault(m.group(2), []).append(int(m.group(1)))
    names = [f"@{u} ({_numbers(n)})" for u, n in by_user.items()]
    if not names:
        return ""
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"With fixes from {joined}."


def git_credits(repo: Path, version: str) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout
    try:
        prev = git("describe", "--tags", "--abbrev=0", f"v{version}^").strip()
        subjects = git("log", "--merges", "--reverse", "--format=%s", f"{prev}..v{version}").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"warning: no credits from git ({exc})", file=sys.stderr)
        return ""
    return credits_from_subjects(subjects)


def render_article(version: str, hl: dict, credits: str) -> str:
    title = re.sub(r"\{([^}]+)\}", r'<span class="hl c-yellow">\1</span>', inline(hl["title"]))
    date = hl["date"] or dt.date.today().strftime("%b %Y")
    fine = " ".join(x for x in (inline(hl["fine"]), credits) if x)
    body = [f"            <h3>{title}</h3>"] + [f"            <p>{inline(p)}</p>" for p in hl["paras"]]
    if fine:
        body.append(f'            <p class="fine">{fine}</p>')
    return "\n".join([
        f"        {LEAD}",
        f'          <div class="release-tag">{version}<small>{date}</small></div>',
        "          <div>", *body, "          </div>",
        "        </article>", "", "",
    ])


def insert_release(index: str, version: str, article: str) -> str:
    if f'<div class="release-tag">{version}<small>' in index:
        print(f"index.html: {version} is already in What's new, not adding it again")
        return index
    if index.count(LEAD) != 1 or index.count(RELEASES) != 1:
        raise SiteError("index.html: expected one releases list and one lead release")
    index = index.replace(LEAD, '<article class="release">')
    return index.replace(RELEASES, RELEASES + article)


def cost_values(report: dict) -> dict[str, tuple[int, int | None]]:
    mcp = {r["profile"]: r for r in report["mcp"]}
    default = next(r for p, r in mcp.items() if p.startswith("(unset"))
    return {
        "all": (mcp["all"]["tokens"], mcp["all"]["tools"]),
        "default": (default["tokens"], default["tools"]),
        "skill_body": (report["skill"]["skill_total"], None),
        "skill_frontmatter": (report["skill"]["frontmatter"], None),
    }


def _width(value: int, top: int) -> str:
    w = value / top * 100
    return "100" if w >= 99.95 else f"{w:.1f}" if w >= 1 else f"{w:.2f}"


def apply_costs(pages: dict[str, str], report: dict) -> None:
    values = cost_values(report)
    top = values["all"][0]
    for key, (tokens, tools) in values.items():
        pat = rf'<li class="bar" data-cost="{key}">.*?</li>'
        m = re.search(pat, pages["index.html"])
        if not m:
            raise SiteError(f'index.html: no <li class="bar" data-cost="{key}">')
        li = re.sub(r"--w:[\d.]+%", f"--w:{_width(tokens, top)}%", m.group(0))
        li = re.sub(r'(class="bar-val">)[\d,]+<', rf"\g<1>{tokens:,}<", li)
        if tools is not None:
            li = re.sub(r"<small>\d+ tools", f"<small>{tools} tools", li)
        pages["index.html"] = pages["index.html"][: m.start()] + li + pages["index.html"][m.end():]
    spans = 0
    for name in pages:
        def repl(m: re.Match) -> str:
            return f'{m.group(1)}{values[m.group(2)][0]:,}</span>'
        pages[name], n = re.subn(r'(<span data-cost="(\w+)">)[\d,]+</span>', repl, pages[name])
        spans += n
    if not spans:
        raise SiteError('no <span data-cost="..."> anywhere; the prose numbers would go stale')


def measure(repo: Path) -> dict:
    out = subprocess.run([sys.executable, str(repo / "scripts" / "measure_context_cost.py"), "--json"],
                         cwd=repo, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True, help="X.Y.Z, without the v")
    ap.add_argument("--site", required=True, type=Path, help="gh-pages checkout")
    ap.add_argument("--highlights", type=Path, help="site/releases/X.Y.Z.md")
    ap.add_argument("--costs", type=Path, help="measure_context_cost.py --json output (default: measure now)")
    ap.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        if not re.fullmatch(VER, args.version):
            raise SiteError(f"--version must look like 1.2.3, got {args.version!r}")
        pages = {}
        for name in PAGES:
            path = args.site / name
            if not path.is_file():
                raise SiteError(f"{path} not found; is --site a gh-pages checkout?")
            pages[name] = path.read_text(encoding="utf-8")
        before = dict(pages)
        bump_versions(pages, args.version)
        if args.highlights:
            hl = parse_highlights(args.highlights.read_text(encoding="utf-8"))
            credits = "" if "@" in hl["fine"] else git_credits(args.repo, args.version)
            pages["index.html"] = insert_release(pages["index.html"], args.version, render_article(args.version, hl, credits))
        report = json.loads(args.costs.read_text()) if args.costs else measure(args.repo)
        apply_costs(pages, report)
    except SiteError as exc:
        print(f"site_release: {exc}", file=sys.stderr)
        return 1
    for name, text in pages.items():
        if text != before[name]:
            (args.site / name).write_text(text, encoding="utf-8")
            print(f"updated {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
