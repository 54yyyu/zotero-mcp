"""Search variants come in a fixed order, whatever the interpreter's hash seed.

``_generate_search_variants`` used to collect the variants in a ``set`` and
return ``list(set)``. String hashing is randomised per process, so the order
changed from one server start to the next — and the API search path lists
each variant's hits in variant order, so one query ranked its results
differently depending on when the server had been started.
"""

import os
import subprocess
import sys
from pathlib import Path

from zotero_mcp import utils as _utils

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_query_as_typed_comes_first():
    assert _utils._generate_search_variants("Cladder Micus") == ["Cladder Micus", "Cladder-Micus"]
    assert _utils._generate_search_variants("Müller-Schmidt") == [
        "Müller-Schmidt",
        "Muller-Schmidt",
        "Müller Schmidt",
        "Muller Schmidt",
        "Mueller-Schmidt",
    ]


def test_order_does_not_depend_on_the_hash_seed():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {SRC!r})\n"
        "from zotero_mcp.utils import _generate_search_variants\n"
        "print(ascii(_generate_search_variants('Müller-Schmidt Björk Straße')))\n"
    )
    outputs = set()
    for seed in ("0", "1", "2", "3", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1, outputs
