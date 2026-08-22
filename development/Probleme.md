# Problems

Bugs and infrastructure issues found while building `stochpylib`, how they were fixed
(or why they're still open), a severity rating (1 = cosmetic, 10 = critical /
wrong-numbers-silently-shipped), and a status. Ordered by entry number (oldest first);
the numbering is continuous with no gaps. Entry #12 resolves the open remainder of #9,
and #10 documents two bugs that were caught during development and never shipped —
both are kept for the lessons they encode.

**Status legend:**
- 🟢 `fixed` — code changed and verified (by tests against independent oracles,
  statistical assertions with fixed seeds, or direct inspection); nothing pending.
- 🟡 `partial` — a fix shipped but verification is incomplete, or a real known caveat
  remains.
- 🔴 `closed` — no code fix applied: declined/won't-fix, a non-goal, or moot.

Every entry follows: **Problem** (what was wrong) → **Fix** (what changed) →
**Verification** (how it was confirmed). This project's standing verification convention:
library code never wraps `scipy.stats`; instead `scipy.stats` is the *test oracle*, so most
numerical fixes are verified by cross-checks against it plus deterministic statistical
assertions (fixed seeds, tolerances ≥ 3 standard errors).

---

### 1. `independence.py` doctest/test examples were mathematically wrong

**Severity:** 4/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** The docstring examples for `is_independent()`, `pairwise_independence()` and
`conditional_independence()` used event pairs that happened to be independent on a
4-outcome uniform space while the docstrings/tests asserted dependence (`False`). The
function logic was correct throughout — only the hand-picked example data was wrong.

**Fix:** Replaced the example event sets with genuinely dependent ones
(`event(1, 2)` vs `event(1, 2, 3)`).

**Verification:**
- Originally surfaced as 2 hard doctest failures on first `pytest` run; after the fix the
  examples were re-checked by hand and the suite went green.

---

### 2. `derangement()` used floating-point summation despite the "exact arithmetic" claim

**Severity:** 3/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** First implementation computed D(n) via the floating-point series
`n! * sum((-1)^k / k!)` then rounded. Float64 precision runs out well before `n!`
overflows, so large inputs would silently produce a wrong integer — contradicting the
module's exact-arithmetic promise.

**Fix:** Rewrote using the exact integer recurrence
`D(n) = (n-1) * (D(n-1) + D(n-2))`, `D(0)=1`, `D(1)=0`.

**Verification:**
- No floating point remains in the function (inspection); the derangement unit tests and
  the module's exact-arithmetic doctests pass for small and large `n`.

---

### 3. Colocated `tests.py` shipped inside the built wheel despite an exclusion rule

**Severity:** 3/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** `[tool.setuptools.exclude-package-data]` only filters *data* files; regular
`.py` modules are always packaged. The exclusion rule was a silent no-op and
`python -m build` shipped `stochpylib/probability/tests.py` to end users.

**Fix:** Relocated tests out of the package entirely to `tests/<module>/tests.py` at the
repo root, making the problem moot rather than patching around setuptools semantics.

**Verification:**
- Wheel contents inspected before and after relocation (`python -m build` + archive
  listing): no test modules ship anymore; this layout is now the documented convention
  (see `tests/README.md`).

---

### 4. Vault generator scripts referenced an unrelated prior project

**Severity:** 4/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** `generate_code_graph.py`'s `IGNORE_DIRS` listed another project's vault
folder name, so running it would have walked into this repo's own vault and generated
bogus "code" notes for Markdown files. `regenerate_vault.py` hardcoded a wrong title and
skipped "the vault" by comparing directory *names* instead of resolved paths — which
never matched.

**Fix:** Corrected `IGNORE_DIRS` to the real vault folder name plus build/cache dirs;
fixed the title and switched the skip check to resolved-path comparison.

**Verification:**
- Both scripts ran end-to-end; `Project-Map.md` correctly excludes the vault and carries
  the right title.

---

### 5. `GPareto.pdf` leaked probability below its support

**Severity:** 6/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** For `shape > 0` the pdf formula was masked only by `t > 0` (with
`t = 1 + shape*z`), not by support membership `z >= 0`. Points *below* `loc` returned
large positive densities — mass was created outside the support.

**Fix:** Mask extended to `(z >= 0) & (t > 0)`.

**Verification:**
- Caught by the scipy cross-check audit (`pdf(-1)`: ours 4.69 vs reference 0.0); the
  dedicated `test_gpareto_support_mask` regression test now asserts exactly this case,
  and the full scipy comparison passes at rtol 1e-6.

---

### 6. `Rice.pdf` overflowed to NaN for large x, poisoning numeric moments

**Severity:** 5/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** The density factored `i0e(y) * exp(y)` to undo the Bessel exponential
scaling. The factors cancel analytically but overflow numerically (`exp(y)` → inf for
y > ~709 while the Gaussian factor underflows), producing NaN that quadrature probes in
generic moment fallbacks picked up — `skewness()` returned NaN despite Rice having
finite moments of all orders.

**Fix:** Pdf computed in log space via `log I0(y) = y + log(i0e(y))`.

**Verification:**
- Matches scipy.stats.rice to machine precision at test points including x = 800
  (previously NaN); interface-contract `skewness()` now agrees with scipy's to 1e-12.

---

### 7. Discrete `ppf` overshot bounded supports and returned the wrong atom

**Severity:** 5/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** `_ppf_discrete`'s bracket-expansion loop returned `high` as soon as the
expanding index crossed the support bound — before the binary search could find the true
smallest atom. E.g. `DiscreteUniform(0, 9).ppf(0.9)` returned 9 although `cdf(8) = 0.9`.

**Fix:** Expansion clamps to `high` and falls through to the binary search, tracking the
last known-below bracket.

**Verification:**
- The scipy cross-check suite compares `ppf` at q ∈ {0.2, 0.55, 0.9} for every bounded-
  support discrete class (DiscreteUniform, Binomial, BetaBinomial, Hypergeometric) —
  all match exactly.

---

### 8. `MultivariateDistribution.fit` had a broken instance-method signature

**Severity:** 2/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** Declared `def fit(cls, data): raise NotImplementedError` without
`@classmethod`, so instance calls raised a confusing `TypeError` instead of the intended
clear error.

**Fix:** Decorated with `@classmethod`.

**Verification:**
- Latent only (every current multivariate class overrides `fit` properly); confirmed the
  base-class call path now raises `NotImplementedError` and the multivariate convention
  tests still pass.

---

### 9. `StableDistribution.rvs` unusably slow; exact special cases routed through numerics

**Severity:** 4/10 · **Status:** 🟢 `fixed` (2dfc156, corner completed by #12)

**Problem:** Generic inverse-CDF sampling evaluated the Gil-Pelaez numerical CDF once per
brentq iteration per sample (each evaluation a full Fourier-inversion quadrature), making
`rvs(300)` take minutes. alpha=2 (always Gaussian) and alpha=1/beta=0 (Cauchy) were also
needlessly routed through numerics.

**Fix:** alpha=2 delegates exactly to Gaussian closed forms, alpha=1/beta=0 to Cauchy;
all alpha != 1 use a Chambers–Mallows–Leckie sampler whose constants were determined
empirically by matching our closed-form characteristic function at Monte-Carlo noise
level across symmetric and skewed parameter sets.

**Verification:**
- `test_stable_alpha_two_delegates_to_gaussian` and `test_stable_cauchy_special_case`
  assert exact equality with scipy closed forms; the CML sampler cf-match test locks the
  validated constants in (max deviation < 0.02 ≈ MC noise).
- The remaining alpha=1/beta≠0 corner is tracked separately and resolved in #12.

---

### 10. Base-2 digital-net engine: two construction bugs caught by exactness checks

**Severity:** 7/10 · **Status:** 🟢 `fixed` (never shipped — caught pre-release, 2dfc156)

**Problem:** (a) Dimension 1 was routed through the generic direction-number recurrence
for its degree-1 polynomial (x+1); the recurrence doubles even integers there and
corrupts the stream — dimension 1 must be plain van der Corput with m_j = 1 at every
level. (b) Point generation used a single-XOR Gray walk on the natural index counter;
that enumerates points in *Gray-code order*, not natural order (a natural-order increment
flips many bits at once), so dim-1 output did not equal van der Corput in sequence
position.

**Fix:** Row 0 short-circuits to pure van der Corput; generation replaced with direct bit
decomposition over natural indices (`x_i = XOR of V[:, b] over the set bits b of i`),
vectorized.

**Verification:**
- Dim-1 equals van der Corput bitwise over the first 16 points; the canonical 2-D prefix
  ((1/2,1/2), (1/4,3/4), (3/4,1/4)) matches published tables; both locked in by tests.

---

### 11. Base-2 nets were not certified exact (t,m,s)-nets under simple initial values

**Severity:** 2/10 · **Status:** 🟢 `fixed` (unreleased — ships with next tag)

**Problem:** Direction numbers came from GF(2) polynomials with simple canonical odd
initial values rather than published optimized tables. Half-interval balance in
dimensions ≥ 2 could be off by ±1 sample at n = 2^m. Root-caused later: part of the
imbalance belongs to any origin-skipped streaming window, but the underlying net also
wasn't certified.

**Fix:** Embedded the standard Joe–Kuo lineage direction-number table
(`_direction_numbers.py`, 64 dims × 30 columns — the same data scipy.stats.qmc uses).
It was extracted from the oracle via the `x_{2^b} = v_b` identity and verified dyadic
plus bitwise round-trip *before* being embedded. `SobolSequence` uses it by default; new
`generate_block(m)` returns the aligned first-2^m block **including the origin**, which
is exactly balanced in every dimension. Streaming `generate(n)` still skips the origin,
so its windows remain ±1 misaligned by construction (documented). GF(2) machinery
remains as fallback beyond dimension 64 and for custom/Niederreiter nets.

**Verification:**
- `test_generate_block_exact_net_balance`: perfect halves/eighths in every dimension at
  n = 2^8 and 2^10.
- `test_block_matches_scipy_set_and_gray_order`: block set-identical to scipy's and
  equal under the Gray-code position mapping.
- Pre-embedding checks: every column dyadic, column 0 identically 1/2.

---

### 12. StableDistribution alpha=1, beta≠0 sampling was unusably slow

**Severity:** 4/10 · **Status:** 🟢 `fixed` (unreleased — ships with next tag)

**Problem:** The alpha=1 skewed corner fell back to per-sample numerical root finding on
the Gil-Pelaez CDF (~seconds for a handful of draws). Two attempts at adopting a
published closed-form Chambers–Mallows–Leckie variant failed validation against our own
characteristic function (best residual 0.077 vs noise level ~0.003 across twelve
structural variants; shift-fitting proved structural mismatch, not a location offset).

**Fix:** Cached numerical quantile table per parameter set: the exact Gil-Pelaez CDF is
evaluated on a grid inside the numerically reliable window (|x − loc| ≤ 25σ), refined
through monotone PCHIP, with the exact power-law tail asymptotics of alpha=1 stables
(`1-F(x) ~ c(1+β)/(πx)` as x → ∞, symmetric on the left) grafted on beyond it down to
q = 1e-9. Draws are a vectorized table lookup. Warmup ~5 s per parameter set
(class-level cache), afterwards O(1) per draw.

**Verification:**
- Empirical characteristic function of 20k draws matches the closed form within 0.012
  (MC noise level) for β = ±{0.5, 0.7}.
- Central quantile error vs root-refined truth ≤ ~1e-3 · scale (asserted over random q).
- Seeded determinism asserted; cache reuse makes repeat instances instant.
