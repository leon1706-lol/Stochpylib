# Problems — Bugs Found During Implementation

Audit log of bugs or issues actually found while implementing/testing `stochpylib`, not a general
changelog (that's [CHANGELOG.md](CHANGELOG.md)). Severity is 1 (trivial) – 10 (critical/data-loss).

### [1] `independence.py` doctest/test examples were mathematically wrong

- **File:** `stochpylib/probability/independence.py`, `tests/probability/tests.py`
- **Problem:** The original docstring examples for `is_independent()`, `pairwise_independence()`,
  and `conditional_independence()` used event pairs that were actually independent (or, for the
  conditional case, actually conditionally independent), but the docstrings/tests asserted the
  opposite (`False`). The underlying function logic was correct throughout — only the hand-picked
  example data was wrong.
- **Impact:** `pytest` failed on first run (2 failures) with a clear mismatch between expected and
  actual doctest output; would have shipped misleading documentation examples if not caught.
- **Fix:** Replaced the example event sets with ones that are genuinely dependent
  (`event(1, 2)` vs `event(1, 2, 3)` instead of two sets that happen to satisfy the independence
  equation by coincidence on a 4-outcome uniform space), re-verified by hand and by the test suite.
- **Severity:** 4/10
- **Status:** Fixed

### [2] `derangement()` initially used floating-point summation despite "exact arithmetic" claim

- **File:** `stochpylib/probability/combinatorics.py`
- **Problem:** First implementation computed D(n) via the floating-point series
  `n! * sum((-1)^k / k!)` then `round()`-ed the result. The module docstring promises exact
  arbitrary-precision integer arithmetic throughout; floating-point summation can lose precision
  for large `n`, silently producing a wrong integer after rounding.
- **Impact:** Latent — not yet manifested in any test (`n` tested was small), but would surface
  as silently wrong results for larger inputs since float64 mantissa precision runs out well
  before `n!` does.
- **Fix:** Rewrote using the exact integer recurrence `D(n) = (n-1) * (D(n-1) + D(n-2))`,
  `D(0)=1`, `D(1)=0` — no floating point anywhere in the function.
- **Severity:** 3/10
- **Status:** Fixed

### [3] Colocated `tests.py` was shipped inside the built wheel despite an exclusion rule

- **File:** `pyproject.toml` (`[tool.setuptools.exclude-package-data]`)
- **Problem:** `exclude-package-data` only filters non-code data files matched via
  `package_data`/`include_package_data`; it does not exclude regular `.py` source modules that
  setuptools picks up as part of a package. The rule was a silent no-op — `python -m build`
  still added `stochpylib/probability/tests.py` to the wheel.
- **Impact:** Shipped a `pytest`-dependent test module to end users inside the installed package
  (harmless functionally, but bloats the wheel and leaks a dev-only import).
- **Fix:** Relocated tests out of the package entirely, to `tests/<module>/tests.py` at the repo
  root — this makes the exclusion problem moot rather than patching around it.
- **Severity:** 3/10
- **Status:** Fixed

### [4] Vault generator scripts still referenced an unrelated prior project and the wrong vault folder name

- **File:** `Stochpylib-Obsidian-Vault/scripts/generate_code_graph.py`,
  `Stochpylib-Obsidian-Vault/scripts/regenerate_vault.py`
- **Problem:** `generate_code_graph.py`'s `IGNORE_DIRS` listed `Aether-vault-Obsidian-Vault`
  (a different project's vault folder name) instead of this repo's actual
  `Stochpylib-Obsidian-Vault`, so running it would have walked into the vault itself and
  generated bogus "code" notes for vault Markdown/scripts. `regenerate_vault.py` hardcoded
  `# Project Map — Aether-Vault` as the title and skipped "the vault" by comparing directory
  *names* (`Path(__file__).parent.name`, i.e. `scripts`) rather than the vault path, which never
  actually matched a top-level repo entry.
- **Impact:** Running either script as documented in `Essential-Tasks.md` would have produced
  incorrect output (wrong title, vault content polluting `code/` notes) the first time anyone
  actually ran them.
- **Fix:** Updated `IGNORE_DIRS` to the real vault folder name plus build/cache directories;
  fixed `regenerate_vault.py`'s title and its vault-skip check to compare resolved paths instead
  of directory names. Verified by running both scripts end-to-end — `Project-Map.md` now correctly
  excludes the vault and reads `# Project Map — stochpylib`.
- **Severity:** 4/10
- **Status:** Fixed
