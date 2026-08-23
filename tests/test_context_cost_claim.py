"""The CLI-is-cheaper claim stays true as both sides change.

The README and CHANGELOG say the CLI + skill route costs far less fixed
context than the MCP tool surface. That claim is only worth making while it
holds, and both sides drift: tools get added, the skill gets longer. These
assert the *relationship* rather than exact counts, so ordinary edits don't
fail the build but a change that invalidates the claim does.

Run `python scripts/measure_context_cost.py` for the current numbers.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tiktoken")

REPO = Path(__file__).resolve().parent.parent
GENERATOR = REPO / "scripts" / "measure_context_cost.py"


@pytest.fixture(scope="module")
def report():
    """Measure in a subprocess.

    Comparing toolset profiles requires re-importing the package under
    different environment variables, which means evicting it from sys.modules.
    Doing that inside the test process leaves every other test module holding
    a stale reference -- their monkeypatches then land on modules nothing
    uses. A subprocess makes that structurally impossible rather than
    something to remember.
    """
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def default_profile(report):
    return next(r for r in report["mcp"] if r["profile"].startswith("(unset"))


class TestMeasurementIsReal:
    def test_profiles_differ(self, report):
        """If every profile reports the same surface, the measurement is
        reading the *registered* tools rather than the *enabled* ones, and the
        whole comparison is meaningless. This caught exactly that."""
        totals = {r["profile"]: r["tokens"] for r in report["mcp"]}
        assert len(set(totals.values())) == len(totals), (
            f"toolset profiles should differ in cost, got {totals}"
        )

    def test_profiles_are_ordered_as_their_names_imply(self, report):
        by_profile = {r["profile"]: r for r in report["mcp"]}
        core_only = by_profile["none"]["tokens"]
        default = next(v for k, v in by_profile.items() if k.startswith("(unset"))["tokens"]
        everything = by_profile["all"]["tokens"]
        assert core_only < default < everything

    def test_tool_counts_are_plausible(self, default_profile):
        assert 20 < default_profile["tools"] < 80

    def test_cost_includes_parameter_schemas(self, default_profile):
        """Descriptions alone would understate the surface by a wide margin --
        the JSON schemas are most of what goes over the wire."""
        heaviest = max(default_profile["per_tool"].values())
        assert heaviest > 200, "per-tool cost looks like descriptions only"


class TestTheClaim:
    def test_skill_frontmatter_is_a_rounding_error_next_to_the_tool_surface(
        self, report, default_profile
    ):
        """This is the headline: until the skill fires, the CLI route puts
        almost nothing in context."""
        frontmatter = report["skill"]["frontmatter"]
        assert frontmatter * 20 < default_profile["tokens"], (
            f"skill frontmatter ({frontmatter}) should be far under 1/20th of "
            f"the MCP default surface ({default_profile['tokens']})"
        )

    def test_the_whole_skill_still_costs_less_than_the_tool_surface(
        self, report, default_profile
    ):
        """A skill that grew past the surface it replaces would have lost the
        argument, however good its advice."""
        assert report["skill"]["skill_total"] < default_profile["tokens"]

    def test_even_the_reference_included_stays_cheaper(self, report, default_profile):
        worst = report["comparison"]["cli_worst_case_with_reference"]
        assert worst < default_profile["tokens"], (
            f"skill + reference ({worst}) has grown past the MCP default "
            f"surface ({default_profile['tokens']}) -- the claim no longer holds"
        )


class TestReportShape:
    def test_json_report_carries_what_the_docs_quote(self, report):
        c = report["comparison"]
        for key in ("mcp_default_always_loaded", "cli_always_loaded",
                    "cli_after_skill_fires", "cli_worst_case_with_reference"):
            assert isinstance(c[key], int) and c[key] > 0

    def test_encoding_is_recorded(self, report):
        """A token count without its tokenizer is not a measurement."""
        assert report["encoding"]
