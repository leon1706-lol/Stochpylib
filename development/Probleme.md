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

### [5] `GPareto.pdf` leaked probability below its support (shape != 0 branch)

- **File:** `stochpylib/distributions/continuous.py`
- **Problem:** For `shape > 0` the pdf formula `t**(-1/shape - 1)/scale` with `t = 1 + shape*z`
  was only masked by `t > 0`, not by membership of the support (`z >= 0`). Points *below* `loc`
  (e.g. `x = -1` for `loc=0`) have `0 < t < 1` and returned a large positive density — mass was
  created outside the support and the pdf did not integrate to 1.
- **Impact:** Silent wrong densities below the support; caught by the scipy cross-check audit
  (`ours=4.69` vs `ref=0.0` at `x=-1`).
- **Fix:** Mask is now `(z >= 0) & (t > 0)`.
- **Severity:** 6/10
- **Status:** Fixed

### [6] `Rice.pdf` overflowed to NaN for large x, poisoning numeric moments

- **File:** `stochpylib/distributions/continuous.py`
- **Problem:** The density factored `... * special.i0e(y) * np.exp(y)` (with `y = x*nu/sigma**2`)
  to undo the exponential scaling of the Bessel function. The factors cancel analytically, but
  numerically `exp(y)` overflows to `inf` for `y > ~709` while the Gaussian factor underflows to
  `0`, producing `NaN`. Quadrature probes in generic moment/entropy fallbacks wander far into the
  tail, so `skewness()` returned NaN even though Rice has finite moments of all orders.
- **Impact:** Wrong/NaN moments from otherwise-correct generic machinery; latent until the
  interface-contract tests exercised `skewness()`.
- **Fix:** Pdf is now computed in log space using `log I0(y) = y + log(i0e(y))`; matches scipy to
  machine precision and never overflows.
- **Severity:** 5/10
- **Status:** Fixed

### [7] Discrete `ppf` overshot bounded supports and returned the wrong atom

- **File:** `stochpylib/distributions/_base.py` (`_ppf_discrete`)
- **Problem:** The bracket-expansion loop returned `high` as soon as the expanding index crossed
  the support's upper bound — before the binary search could locate the true smallest atom `k`
  with `cdf(k) >= q`. E.g. `DiscreteUniform(0, 9).ppf(0.9)` returned 9 although the correct atom
  is 8 (`cdf(8) = 0.9`). Only bounded-support discrete distributions were affected, and only when
  the expansion step overshot past `high`.
- **Impact:** Wrong quantiles on boundary probabilities; generic inverse-CDF sampling over-weighted
  the endpoint near boundary quantiles.
- **Fix:** Expansion now clamps to `high` and falls through to the binary search, tracking the
  last known-below bracket; verified against scipy for all bounded-support discrete classes.
- **Severity:** 5/10
- **Status:** Fixed

### [8] `MultivariateDistribution.fit` had a broken instance-method signature

- **File:** `stochpylib/distributions/_base.py`
- **Problem:** Declared as `def fit(cls, data): raise NotImplementedError` without
  `@classmethod` — an instance call `d.fit(data)` raised a confusing `TypeError` (missing
  argument) instead of the intended `NotImplementedError`.
- **Impact:** Latent only — every current multivariate class overrides `fit` properly.
- **Fix:** Decorated with `@classmethod`.
- **Severity:** 2/10
- **Status:** Fixed

### [9] `StableDistribution.rvs` unusably slow; special cases routed through numerics

- **File:** `stochpylib/distributions/heavy_tail.py`
- **Problem:** Generic inverse-CDF sampling evaluated the Gil-Pelaez numerical CDF once per
  brentq iteration per sample — each evaluation being a full Fourier-inversion quadrature —
  making `rvs(300)` take minutes. The exactly-solvable cases (alpha=2 Gaussian, alpha=1/beta=0
  Cauchy) were also needlessly routed through numerics.
- **Impact:** Performance made the spec-required `.rvs()` practically unusable for general
  stables; special cases slower and less exact than necessary.
- **Fix:** alpha=2 (any beta) now delegates exactly to Gaussian closed forms, alpha=1,beta=0 to
  Cauchy. All alpha != 1 use a Chambers–Mallows–Leckie sampler whose constants were determined
  empirically by matching the closed-form characteristic function at Monte-Carlo noise level
  across symmetric and skewed parameter sets (locked in by tests).
- **Open remainder:** No validated closed-form sampler found for `alpha == 1, beta != 0`; that
  corner intentionally keeps the slow-but-correct inverse-CDF fallback (documented in the
  docstring).
- **Severity:** 4/10
- **Status:** Partially fixed (open item: fast sampler for alpha=1, beta!=0)

### [10] Base-2 digital-net engine: two construction bugs caught by exactness checks

- **File:** \stochpylib/montecarlo/quasi_random.py\ (caught during development, never shipped)
- **Problem:** (a) Dimension 1 was routed through the generic direction-number recurrence for
  its degree-1 polynomial (x+1); the recurrence doubles even integers there and corrupts the
  stream - dimension 1 must be plain van der Corput with m_j = 1 at every level. (b) Point
  generation used an Antonov-Saleev-style single-XOR Gray walk on the natural index counter;
  such a walk enumerates the points in *Gray-code order*, not natural order, so dim-1 output
  did not equal van der Corput in sequence position (x_i must be the XOR of direction numbers
  over the set bits of i itself; a natural-order increment flips many bits at once).
- **Impact:** Wrong point ordering for the flagship sequence class; would have silently
  degraded every downstream QMC estimator's convergence guarantees.
- **Fix:** Row 0 short-circuits to pure van der Corput; generation replaced with direct bit
  decomposition over natural indices (vectorized). Dim-1 now equals van der Corput bitwise and
  the first three 2-D points match canonical Sobol tables.
- **Severity:** 7/10
- **Status:** Fixed

### [11] Base-2 nets are not certified exact (t,m,s)-nets under simple initial values

- **File:** \stochpylib/montecarlo/quasi_random.py\
- **Problem:** Direction numbers use primitive/irreducible GF(2) polynomials with simple
  canonical odd initial values instead of published optimized tables (Joe-Kuo / Bratley-Fox).
  Measured consequence: half-interval balance in dimensions >= 2 can be off by +-1 sample at
  n = 2^m (dimension 1 is exact). Statistical quality is unaffected in practice (KS p ~ 1,
  discrepancy ~50x better than pseudo-random), but strict net certification is not claimed.
- **Impact:** Cosmetic-to-minor for estimation accuracy; matters only if someone needs the
  formal net property or bitwise-standard tables.
- **Fix:** Documented precisely in the module docstring and module spec; tests assert the
  honest contract (exact dim 1, +-1 elsewhere, uniformity + discrepancy thresholds).
- **Open item:** Adopt published direction-number tables to certify exactness and match
  standard sequences bitwise.
- **Severity:** 2/10
- **Status:** Open (documented limitation)
