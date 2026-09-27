"""Recomputes development/stats.json from live repo state.

This is the single source of truth the README's dynamic shields.io badges read from
raw.githubusercontent.com (see the "tests" and "public names" badges at the top of
README.md) -- so those badges update themselves the moment this file changes, with no
manual README edit. Run automatically by the update-stats-badge CI job after every push
to main (only once every other job has gone green); safe to run locally too.

Numbers are recomputed from the same ground truth tests/docs/tests.py checks against
(pytest collection, tests/library/_spec_names.json, development/Implementation-Checklist.md),
not copied from anywhere -- if that suite is green, this script's output matches it exactly.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATS = REPO / "development" / "stats.json"
SPEC_JSON = REPO / "tests" / "library" / "_spec_names.json"
CHECKLIST = REPO / "development" / "Implementation-Checklist.md"

sys.path.insert(0, str(REPO))
from tests.docs.tests import KNOWN_CONDITIONAL_SKIPS  # noqa: E402


def _collected_test_count():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    match = re.search(r"(\d+) tests collected", out.stdout)
    assert match, f"could not parse collection count:\n{out.stdout[-2000:]}"
    return int(match.group(1))


def _spec_names_total():
    spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
    return sum(len(v) + (13 if k == "distributions" else 0) for k, v in spec.items())


def _spec_names_done():
    text = CHECKLIST.read_text(encoding="utf-8")
    sections = re.findall(r"^## \[[a-z_]+\][^\n]*\((\d+)/(\d+)\)", text, re.MULTILINE)
    assert sections, "could not parse Implementation-Checklist.md sections"
    return sum(int(done) for done, _ in sections)


def main():
    import stochpylib

    collected = _collected_test_count()
    passed = collected - KNOWN_CONDITIONAL_SKIPS
    spec_total = _spec_names_total()
    spec_done = _spec_names_done()

    stats = {
        "version": stochpylib.__version__,
        "tests_collected": collected,
        "tests_passed": passed,
        "tests_skipped": KNOWN_CONDITIONAL_SKIPS,
        "tests_badge": f"{passed} of {collected} passing",
        "spec_names_total": spec_total,
        "spec_names_done": spec_done,
        "spec_names_badge": f"{spec_done} of {spec_total}",
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
