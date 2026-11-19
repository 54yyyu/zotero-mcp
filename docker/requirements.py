"""Print the requirements of the project and the chosen extras, one per line.

The Dockerfile installs these before it copies the source tree, so the layer
holding the dependencies is rebuilt only when pyproject.toml changes.

    python requirements.py pyproject.toml [extra,extra...]
"""

import re
import sys
import tomllib


def requirements(project: dict, extras: list[str]) -> list[str]:
    name = re.sub(r"[-_.]+", "-", project["name"]).lower()
    optional = project.get("optional-dependencies", {})
    found = list(project.get("dependencies", []))
    pending, seen = list(extras), set()
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for requirement in optional[extra]:
            # An extra that names the project itself pulls in other extras.
            own = re.fullmatch(r"\s*([A-Za-z0-9._-]+)\s*\[([^\]]*)\]\s*", requirement)
            if own and re.sub(r"[-_.]+", "-", own.group(1)).lower() == name:
                pending.extend(e.strip() for e in own.group(2).split(",") if e.strip())
            else:
                found.append(requirement)
    return found


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        project = tomllib.load(f)["project"]
    chosen = [e.strip() for e in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if e.strip()]
    print("\n".join(requirements(project, chosen)))
