# stochpylib.random_matrix

Random matrix theory: the classical Gaussian ensembles (GOE/GUE/GSE), Wigner and
Wishart/inverse-Wishart matrices, the circular unitary ensemble, a Ginibre-type
i.i.d. ensemble, the limiting spectral laws (Wigner semicircle, Marchenko-Pastur,
Tracy-Widom) as full library distributions, tridiagonal beta-Hermite/Laguerre and
Jacobi ensembles, Haar-distributed orthogonal/unitary/symplectic matrices, and the
spectral statistics that test them (spacing ratios, level repulsion, empirical
spectral distributions, Tracy-Widom edge scaling, Edelman's hard-edge law). 23 public
names across four submodules, natively on numpy/scipy — no wrapper dependencies.

**Status:** implemented & tested (23/23 spec names).

## Files

- `_common.py` — private helpers: seeded RNG, sorted Hermitian eigenvalues, the
  grid/PCHIP machinery behind tabulated laws (`_GridInterp`), the Wigner surmise
  densities/CDFs, and the large-N mean gap-ratio references `<r>` (Atas et al. 2013).
- `ensembles.py` — `MatrixEnsemble` base (`sample`/`samples`/`eigenvalues`/
  `normalized_eigenvalues`/`limit_law`), `GOE`, `GUE`, `GSE` (quaternion self-dual,
  Kramers-doubled eigenvalues returned once), `WignerMatrix` (i.i.d. entries from any
  library `Distribution` or callable), `WishartMatrix` and `InverseWishart` (sampled
  through `distributions.Wishart`/`InverseWishart`), `CUE` (Haar unitary), `MuresanMatrix`
  (Ginibre-type i.i.d. non-Hermitian, real/complex/custom entries) and the `CircularLaw`
  helper it converges to.
- `empirical_spectra.py` — `WignerSemicircle` (closed-form pdf/cdf/moments/mgf/cf, Beta
  sampler), `MarchenkoPastur` (with the atom at zero for `gamma > 1`, tie-aware KS),
  `TracyWidomDistribution(beta=1|2|4)` (Hastings-McLeod Painleve II solved once per
  `beta` and cached), `BetaEnsemble` (Dumitriu-Edelman tridiagonal beta-Hermite /
  beta-Laguerre, any `beta > 0`), `JacobiEnsemble` (MANOVA ensemble with the Wachter
  limit density).
- `random_rotations.py` — `HaarMeasure(group, n)` for `O(n)`/`U(n)`/`Sp(n)` (Gaussian
  QR with Mezzadri's phase correction; quaternionic Gram-Schmidt for `USp(2n)`),
  `RandomOrthogonalMatrix` (optional `det=+1` for `SO(n)`), `RandomUnitaryMatrix`,
  `RandomSymplectic`.
- `statistics.py` — `EigenvalueSpacing` (polynomial unfolding, unfolding-free gap
  ratios, surmise KS comparison, classification), `LevelRepulsion` (MLE of the
  repulsion exponent in the generalized-surmise family, or the small-gap log-log slope),
  `EigenvalueDistribution` (empirical cdf, native Gaussian KDE, moments, KS against any
  law), `LargestEigenvalue` (Tracy-Widom centering/scaling for Hermitian, Wishart and
  beta ensembles; Johnstone-Ma centering for Wishart), `BulkSpectrum` (density, moments, Dyson-Mehta number variance),
  `SpectralEdge` (extremes, soft-edge test, Edelman's `n * lambda_min` hard-edge law
  for square real Wishart).

## Conventions

- **Normalization.** Hermitian Gaussian ensembles have off-diagonal `E|h_ij|^2 = 1`,
  so `eigenvalues / sqrt(n)` converge to `WignerSemicircle(radius=2)`; Wishart
  `eigenvalues / n` converge to `MarchenkoPastur(p / n)`; Ginibre-type
  `eigenvalues / sqrt(n)` fill the unit disk. `normalized_eigenvalues()` applies exactly
  that scaling and `limit_law()` returns the matching law object.
- **Limit laws are distributions.** `WignerSemicircle`, `MarchenkoPastur` and
  `TracyWidomDistribution` subclass `distributions.Distribution` and satisfy the common
  13-method contract; `fit` is a classmethod returning a new instance (moment estimators
  for the first two; Tracy-Widom has no free parameter). Each also offers
  `compare(eigenvalues)` → `statistics.TestResult` (KS) and `histogram_vs_density()` —
  the plotting-free replacement for the design spec's `plot_semicircle()` /
  `plot_vs_empirical()`.
- **Tracy-Widom conventions.** `beta=2` mean −1.7711, `beta=1` mean −1.2065, `beta=4`
  mean −2.3069 (the classical/Bornemann tables). The β-ensemble edge variable
  `(lambda_max / sqrt(beta n / 2) - 2) n^(2/3)` (Ramirez-Rider-Virag) converges to those
  laws for `beta` = 1, 2 directly and, for `beta = 4`, after multiplication by `2^(1/6)`;
  `LargestEigenvalue` applies that factor.
- **Spacing statistics.** The mean adjacent-gap ratio `<r>` is unfolding-free and is the
  primary oracle (Poisson `2 ln 2 - 1`, GOE 0.5307, GUE 0.5996, GSE 0.6744); the Wigner
  surmise comparisons use a degree-7 polynomial unfolding with 5 % edge trimming.
- Every stochastic method takes `random_state=` (anything `np.random.default_rng`
  accepts, including a live `Generator` for streaming draws).
- Library code never imports `scipy.stats`; `scipy.special/integrate/interpolate` are
  the numerical building blocks and `scipy.stats` (`ortho_group`, `unitary_group`,
  `kstest`, ...) is the test suite's independent oracle only.
- **Name collision.** `random_matrix.InverseWishart` (an ensemble) shares its name with
  `distributions.InverseWishart` (a distribution); `spl show InverseWishart` resolves to
  the distribution, which comes first in `stochpylib.__all__` order.
- **`MuresanMatrix`.** The design spec lists this name without defining it; it is
  implemented as the general i.i.d.-entry non-Hermitian (Ginibre-type) ensemble, which
  the spec otherwise lacks, and documented as such in the vault.

## Known limitations

- `TracyWidomDistribution` tables cover `[-10, 8]` (β=4: `[-7.07, 5.66]`) with cdf
  accuracy ~1e-6; `ppf` clips to that range for `q` below ~1e-12 or above `1 - 1e-12`.
- `LargestEigenvalue` implements the soft-edge scaling for Hermitian Gaussian/Wigner
  ensembles, beta-Hermite ensembles with `beta` in {1, 2, 4}, and real Wishart /
  beta-Laguerre(1) with identity covariance; other cases need explicit
  `center`/`scale`/`beta`. Finite-`n` corrections are `O(n^(-2/3))`, so use `n >= 200`
  for a meaningful Tracy-Widom comparison.
- `JacobiEnsemble` supports `beta` in {1, 2} (real/complex Wishart construction); the
  Wachter `limit_density` assumes `n < min(m1, m2)`.
- `SpectralEdge.hard_edge_*` cover the square real Wishart case (`p = n`, identity
  covariance) only — Edelman's closed form is specific to it.
- `HaarMeasure` eigenangles are exactly uniform only for `U(n)`; `O(n)` carries atoms at
  `±1` and `Sp(n)` shows repulsion from `0`/`π` — mathematics, not a sampler defect.
- No plotting: `histogram_vs_density()` returns arrays to plot with any library.

Spec: vault `Modules/random_matrix.md` (private). Tests:
`tests/random_matrix/tests.py` (oracles) and `tests/random_matrix/e2e.py` (API sweep).
