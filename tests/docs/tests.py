"""Documentation-consistency suite: every number the docs claim must match reality.

The root README, package README, development index and Implementation-Checklist
carry live facts (test counts, spec-name counts, versions, module tables). These
tests fail the suite the moment a doc drifts from the actual package — the same
contract Aether-Quant enforces with auto-generated markers, enforced here by
assertions instead of a rewriting tool.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
PKG_README = REPO / "stochpylib" / "README.md"
DEV_README = REPO / "development" / "README.md"
CHECKLIST = REPO / "development" / "Implementation-Checklist.md"
SPEC_JSON = REPO / "tests" / "library" / "_spec_names.json"

# The two permanently-skipped tests are the VonMises and Kumaraswamy scipy
# cross-checks in tests/distributions/tests.py — no direct scipy mapping
# (circular/different conventions), covered by dedicated checks instead. If
# this constant ever needs changing, the suite below fails first and forces
# the one-line update.
KNOWN_CONDITIONAL_SKIPS = 2

IMPLEMENTED = (
    "probability", "distributions", "montecarlo", "timeseries",
    "gaussian_processes", "copulas", "survival", "queueing",
    "information_theory",
)


def _read(path):
    return path.read_text(encoding="utf-8")


def _pytest_collect_count():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    match = re.search(r"(\d+) tests collected", out.stdout)
    assert match, f"could not parse collection count:\n{out.stdout[-2000:]}"
    return int(match.group(1))


def test_readme_badge_matches_collected_test_count():
    collected = _pytest_collect_count()
    expected_passing = collected - KNOWN_CONDITIONAL_SKIPS
    readme = _read(README)
    match = re.search(r"tests-(\d+)%20passing", readme)
    assert match, "README test badge not found"
    assert int(match.group(1)) == expected_passing, (
        f"README badge says {match.group(1)} passing, "
        f"but {collected} collected - {KNOWN_CONDITIONAL_SKIPS} skips = {expected_passing}"
    )


def test_readme_test_suite_section_states_pass_and_skip():
    collected = _pytest_collect_count()
    expected_passing = collected - KNOWN_CONDITIONAL_SKIPS
    readme = _read(README)
    match = re.search(r"\*\*(\d+) passed / (\d+) skipped\*\*", readme)
    assert match, "README 'N passed / M skipped' statement not found"
    assert int(match.group(1)) == expected_passing
    assert int(match.group(2)) == KNOWN_CONDITIONAL_SKIPS


def test_spl_test_check_count_matches_readme():
    out = subprocess.run(
        [sys.executable, "-m", "stochpylib.cli", "--test"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, f"spl --test failed:\n{out.stdout[-2000:]}"
    match = re.search(r"(\d+) checks", out.stdout)
    assert match, f"could not parse selftest check count:\n{out.stdout[-500:]}"
    checks = int(match.group(1))
    for doc in (README, PKG_README, DEV_README.parent / "infrastructure.md",
                REPO / "tests" / "README.md"):
        assert re.search(rf"{checks}[ -]checks?", _read(doc)), (
            f"{doc.name} does not state the live selftest count ({checks} checks)"
        )


def test_version_strings_agree_everywhere():
    import stochpylib
    pyproject = _read(REPO / "pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject.toml version not found"
    assert match.group(1) == stochpylib.__version__, (
        f"pyproject {match.group(1)} != __init__ {stochpylib.__version__}"
    )
    readme = _read(README)
    version_block = re.search(r"```[a-z]*\n\$ spl --version\n(\S+)\n", readme)
    assert version_block, "README spl --version example block not found"
    assert version_block.group(1) == stochpylib.__version__, (
        f"README version example {version_block.group(1)} != {stochpylib.__version__}"
    )


def test_readme_status_table_covers_every_subpackage_with_true_counts():
    spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
    pkg_dir = REPO / "stochpylib"
    subpackages = sorted(
        p.name for p in pkg_dir.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and p.name != "__pycache__"
    )
    assert subpackages == sorted(IMPLEMENTED), "IMPLEMENTED tuple out of sync"
    readme = _read(README)
    for name in subpackages:
        assert (pkg_dir / name / "README.md").exists(), f"{name}/ lacks README.md"
        row = re.search(rf"\| `stochpylib\.{name}` \| (\d+) \|", readme)
        assert row, f"README Current Status table lacks a row for {name}"
        claimed = int(row.group(1))
        if name == "distributions":
            expected = len(spec[name]) + 13  # 47 classes + 13 shared interface methods
        else:
            expected = len(spec[name])
        assert claimed == expected, (
            f"{name}: README claims {claimed} public names, spec has {expected}"
        )


def test_package_readme_table_covers_every_subpackage():
    pkg_dir = REPO / "stochpylib"
    subpackages = sorted(
        p.name for p in pkg_dir.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and p.name != "__pycache__"
    )
    readme = _read(PKG_README)
    for name in subpackages:
        assert re.search(rf"\| `{name}/` \|", readme), (
            f"stochpylib/README.md table lacks {name}/"
        )


def test_development_readme_lists_every_dev_doc():
    dev_dir = REPO / "development"
    docs = sorted(p.name for p in dev_dir.glob("*.md") if p.name != "README.md")
    readme = _read(DEV_README)
    for name in docs:
        assert f"[`{name}`]({name})" in readme, f"development/README.md lacks {name}"


def test_readme_relative_links_resolve():
    readme = _read(README)
    for target in re.findall(r"\]\(([^)#]+?\.md)\)", readme):
        assert not target.startswith("http"), f"unexpected absolute link: {target}"
        assert (REPO / target).exists(), f"README links to missing file: {target}"


def test_readme_toc_anchors_resolve():
    readme = _read(README)
    headings = re.findall(r"^#{2,3} (.+)$", readme, re.MULTILINE)

    def slug(text):
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"\s+", "-", text)

    anchors = {slug(h) for h in headings}
    for anchor in re.findall(r"\]\(#([^)]+)\)", readme):
        assert anchor in anchors, f"README TOC anchor '#{anchor}' has no heading"


def test_no_stale_claims_in_readme():
    readme = _read(README)
    for stale in (r"\b288\b", r"\b287\b", r"\b496\b", r"\b329\b", r"\b257\b",
                  r"\b229\b", r"\b133 checks\b", r"\b130 checks\b",
                  r"spl --version\n0\.1\.0\b", r"\b136 checks\b",
                  r"\b136-check\b", r"seven modules", r"eight modules"):
        assert not re.search(stale, readme), f"stale claim survived: {stale}"


def test_implementation_checklist_progress_line_matches_reality():
    spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
    text = _read(CHECKLIST)
    sections = re.findall(
        r"^## \[([a-z_]+)\][^\n]*\((\d+)/(\d+)\)(.*?)(?=^## |\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    assert sections, "could not parse checklist sections"
    total_done, total_spec = 0, 0
    for name, done, planned, body in sections:
        done, planned = int(done), int(planned)
        total_done += done
        total_spec += planned
        if name in IMPLEMENTED:
            assert done == planned, f"{name} marked implemented but section says {done}/{planned}"
            assert "- [ ]" not in body, f"{name} complete but has unchecked boxes"
    assert total_spec == 794, f"checklist totals {total_spec}, expected 794"
    progress = re.search(r"\*\*Progress: (\d+) / 794 public names implemented\.\*\*", text)
    assert progress, "progress line missing from checklist"
    assert int(progress.group(1)) == total_done, (
        f"progress line says {progress.group(1)}, sections sum to {total_done}"
    )
    expected_total = sum(
        len(spec[n]) + (13 if n == "distributions" else 0) for n in IMPLEMENTED
    )
    assert total_done == expected_total == 317, (
        f"checklist progress {total_done} != true spec total {expected_total}"
    )


def test_probleme_entries_numbered_continuously():
    text = _read(REPO / "development" / "Probleme.md")
    numbers = [int(n) for n in re.findall(r"^### (\d+)\.", text, re.MULTILINE)]
    assert numbers, "no Probleme entries found"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"Probleme numbering has gaps or duplicates: {numbers}"
    )


def test_logo_present_and_referenced():
    assert (REPO / "development" / "logo.png").exists(), "development/logo.png missing"
    assert "development/logo.png" in _read(README), "README banner does not use the logo"


def test_spl_help_invents_every_module():
    out = subprocess.run(
        [sys.executable, "-m", "stochpylib.cli", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0
    for name in IMPLEMENTED:
        assert name in out.stdout, f"spl --help inventory lacks module {name}"
