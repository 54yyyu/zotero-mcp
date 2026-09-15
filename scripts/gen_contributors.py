#!/usr/bin/env python3
"""Regenerate the contributor avatar grid in README.md from GitHub.

Rewrites the block between the ``contributors:start`` and
``contributors:end`` markers with one linked avatar per contributor, most
contributions first. The images are GitHub's own avatar URLs, so the grid
renders on GitHub and on PyPI without a third-party image service.

    python scripts/gen_contributors.py

Needs the GitHub CLI (``gh``) authenticated, or ``GITHUB_TOKEN`` set.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "54yyyu/zotero-mcp"
README = Path(__file__).resolve().parent.parent / "README.md"
START = "<!-- contributors:start -->"
END = "<!-- contributors:end -->"
AVATAR_SIZE = 64   # pixels requested from GitHub (2x the displayed size)
DISPLAY_SIZE = 32


def fetch_contributors() -> list[dict]:
    """All human contributors, most contributions first."""
    people: list[dict] = []
    page = 1
    while True:
        path = f"repos/{REPO}/contributors?per_page=100&page={page}"
        try:
            raw = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            request = urllib.request.Request(f"https://api.github.com/{path}")
            if token := os.environ.get("GITHUB_TOKEN"):
                request.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(request) as response:
                raw = response.read().decode()
        batch = json.loads(raw)
        if not batch:
            return people
        people.extend(p for p in batch if p.get("type") == "User")
        page += 1


def render(people: list[dict]) -> str:
    cells = [
        f'<a href="{p["html_url"]}" title="{p["login"]}">'
        f'<img src="{p["avatar_url"]}&s={AVATAR_SIZE}" width="{DISPLAY_SIZE}" height="{DISPLAY_SIZE}" alt="{p["login"]}"></a>'
        for p in people
    ]
    return "\n".join([START, '<p align="center">', *cells, "</p>", END])


def main() -> int:
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        print(f"README.md has no {START} ... {END} block", file=sys.stderr)
        return 1
    people = fetch_contributors()
    README.write_text(pattern.sub(lambda _m: render(people), text), encoding="utf-8")
    print(f"wrote {len(people)} contributors to {README.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
