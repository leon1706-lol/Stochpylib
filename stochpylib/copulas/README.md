# stochpylib.copulas — dependence modeling

**Status: implemented and tested** (26/26 spec names).

Layout:

- `_base.py` — `BaseCopula`: fluent `.fit(data)` via rank pseudo-observations,
  params on `attr_`, `random_state=` convention, shared
  cdf/sample/density/kendall_tau/tail_dependence/loglik/aic surface.
- `_utils.py` — pseudo-observations U = rank/(n+1), O(n log n) Kendall's tau-b
  (Fenwick inversion counting, verified against scipy.stats), Spearman's rho,
  native Student-t quantile function (incomplete-beta inverse, no scipy.stats).
- `elliptical.py` — `GaussianCopula`, `StudentTCopula` (any dimension).
  Exact CDF via recursive integration over conditionals (chain-rule Schur
  updates; multivariate-t conditionals stay t with nu+1); tau-based rho +
  coarse-grid profile-MLE degrees of freedom; Kanter/Marshall-Olkin-free
  native sampling through numpy generators.
- `archimedean.py` — Clayton/Gumbel/Frank/Joe/AliMikhailHaq (d-dimensional)
  plus BB1/BB7 and Plackett (bivariate). Generator framework: CDF =
  psi(sum(phi(u))), exact bivariate densities via psi'', Genest–MacKay tau,
  closed-form tau inversion where known (Clayton/Gumbel; Frank via Debye D1),
  cached numeric tau(theta) curve elsewhere; Marshall–Olkin fast paths
  (Clayton gamma mixer, Gumbel Kanter stable mixer), else sequential
  conditional inversion through the generator-derivative ratio
  psi'(phi(u)+phi(w))/psi'(phi(u)).
- `empirical.py` — `EmpiricalCopula` (multivariate e.c.d.f., row bootstrap),
  `CheckerboardCopula` (binned masses, multilinear CDF), `BetaCopula`
  (Bernstein-polynomial smoothing, converges to checkerboard as m grows).
- `pair.py` — `PairCopulaConstruction`: family x rotation (0/90/180/270)
  selection by AIC; h-functions verified against dC/dv of each rotated CDF.
- `vine.py` — `CVine`/`DVine`/`RVine` built on one recursive edge structure;
  R-vines use maximum-spanning-tree selection on |tau| among
  proximity-admissible pairs (Disshmann-style heuristic under the simplifying
  assumption). Likelihood replays the stored columns; simulation follows a
  peel-order plan (each variable drawn conditioned on all previously drawn
  ones). `VineStructureSelect` compares structures by AIC; `VineCopula`
  facade matches the spec quickstart.
- `methods.py` — `CopulaFit` (family dispatcher ranked by AIC), `CopulaSample`,
  `kendall_tau`, `spearman_rho`, `tail_dependence` (analytic + threshold
  estimators), `copula_density`, `conditional_copula`.

Conventions: everything native numpy/scipy.special/optimize/integrate — no
scipy.stats anywhere; sampling uses our own distributions machinery where
relevant; every stochastic method takes `random_state=`.

Known limitations (documented deviations): Frank/Joe/AMH sample exactly only
in 2 dimensions (Clayton/Gumbel cover any d via mixing); BB1/BB7/Plackett are
bivariate families by construction; Plackett tau is numeric; BB7 tail limits
are extrapolated numerically; R-vine structure search is the MST heuristic,
not full search; vine CDFs have no closed form (use loglik/AIC); BetaCopula's
sampler is a documented approximation of its conditional.

Spec: vault `Modules/copulas.md` (private). Tests: `tests/copulas/tests.py`.
