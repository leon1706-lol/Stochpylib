# stochpylib.viz

Statistical visualization, native on this library's own numbers. Every `plot_*()`
computes its data by calling straight into the module that owns it (distributions,
timeseries, survival, gaussian_processes, spatial_statistics, advanced_mcmc, statistics,
copulas, random_matrix, ...) rather than re-deriving anything, and returns a `Figure` — a
small scene graph, not a matplotlib object — 35 spec names across five submodules.
`Figure`/`Axes` are documented extras beyond the spec (the scene graph every plot function
returns/composes into).

**Status:** implemented & tested (35/35 spec names).

```python
import numpy as np
from stochpylib.distributions import Gamma
from stochpylib.viz import plot_histogram, plot_qqplot

g = Gamma(shape=3.0, scale=2.0)
data = g.rvs(500, random_state=1)

fig = plot_histogram(data, dist=g, kde=True)
svg = fig.to_svg()                    # native, zero-dependency renderer
fig.save("histogram.svg")

qq = plot_qqplot(data, dist=g)
qq.data["r"]                          # the numbers behind the plot, e.g. QQ correlation

# optional matplotlib backend (lazy import; raises a clear ImportError if not installed)
mpl_fig = fig.to_matplotlib()
fig.save("histogram.png")
```

## Files

- `_figure.py` — the scene graph: `Figure`/`Axes` plus artist dataclasses (`Line`,
  `Scatter`, `Bars`, `Band`, `Step`, `Stem`, `RefLine`, `Segments`, `Heatmap`, `Text`,
  `Arrow`, `Circle`). `Axes.data_limits()` computes auto axis limits with 5% padding;
  `Figure.data` carries the numbers each plot function exposes.
- `_svg.py` — the native renderer: nice-number ticks, per-axes clip rectangles, and one
  SVG element type per artist kind. Zero dependencies, deterministic (same `Figure` ->
  byte-identical SVG), the only rendering path guaranteed to work everywhere.
- `_mpl.py` — the optional matplotlib backend. `matplotlib` is imported lazily, only
  inside this file's function bodies (enforced by a library-wide AST guard) — importing
  `stochpylib.viz` or using the native SVG path never requires it installed.
  `is_available()` checks without importing anything module-level.
- `_ticks.py` — Heckbert "nice numbers" tick placement (linear and log/decade), compact
  number formatting.
- `_colors.py` — a 10-color categorical palette and two continuous colormaps
  (viridis-like sequential, RdBu-like diverging).
- `_common.py` — shared validation/RNG helpers, the distribution plotting grid
  (`ppf([.001, .999])` clipped to support), Filliben plotting positions, Durbin-Levinson
  PACF from an ACF sequence, MCMC sampler/array normalization (`_extract_chains` keeps
  per-chain structure for trace plots; `_extract_samples` flattens to one samples table for
  posterior/pair plots), and OLS diagnostics (hat values, studentized residuals, Cook's D).
- `distributions.py` — `plot_pdf`/`plot_pmf`/`plot_cdf`/`plot_survival`/`plot_hazard`,
  `plot_qqplot`/`plot_ppplot`, `plot_histogram`, `plot_kde`.
- `processes.py` — `plot_process`, `plot_acf`/`plot_pacf`, `plot_periodogram`,
  `plot_spectrogram`, `plot_wavelet` (Morlet CWT or DWT detail levels), `plot_trajectory`.
- `diagnostics.py` — `trace_plot`/`posterior_plot`/`pair_plot` (MCMC), `residual_plot`/
  `leverage_plot`/`influence_plot` (regression), `funnel_plot` (meta-analysis).
- `multivariate.py` — `plot_heatmap`, `plot_correlation`, `plot_copula`,
  `plot_scatter_matrix`, `plot_biplot` (PCA), `plot_dendrogram` (agglomerative linkage).
- `special.py` — `plot_markov_chain`, `plot_brownian`, `plot_gp`, `plot_survival_km`,
  `plot_variogram`, `plot_eigenvalues`.

## Conventions

- **Every function returns a `Figure`.** `ax=None` creates a fresh one; passing an `Axes`
  draws into it and returns its owning `Figure` (the composition contract every function
  follows, so panels can be assembled by hand: `fig.axes[1]` then `plot_acf(x, ax=that_axes)`).
- **`fig.data` carries the plotted numbers**, not just pixels — the same numbers the tests
  assert against (e.g. `plot_pdf(...).data["x"]`/`["pdf"]`, `plot_acf(...).data["acf"]`).
- **SVG is the default; matplotlib is optional.** `Figure.to_svg()`/`.save("*.svg")` never
  need matplotlib. `Figure.to_matplotlib()`/`.save("*.png"/"*.pdf"/...)` do, and raise a
  clear `ImportError` naming the native fallback if it isn't installed.
- **Reuse, not re-derivation.** Every plot's numbers come from an existing stochpylib
  function/class (scipy is never wrapped here either — the reused code already avoids
  that). `experimental_design.InteractionPlot`/`NormalPlot` and `spatial_statistics`'s
  variogram/covariance/summary-function classes gained a `to_figure()` method that lazily
  imports `stochpylib.viz` for exactly this reason.
- **Stochastic functions take `random_state=`.**

## Known limitations

- **No interactivity** — no zoom/pan/hover; this is a static-image renderer (SVG or a
  matplotlib figure), not a plotting widget.
- **No 3-D plots.**
- **SVG text-layout metrics are approximate** — the renderer does not measure glyph
  widths, so long tick/legend labels aren't guaranteed not to overlap in the native
  backend; the matplotlib backend lays text out properly.
- **`plot_wavelet`'s CWT and `plot_eigenvalues`' complex-spectrum path assume the caller
  passes sensible units** (e.g. `fs=1.0`'s "samples per unit" convention from
  `timeseries.spectral`) — scale/frequency axes are in whatever units the input implies.
- **Dense grids (`plot_copula(kind="density")`, `plot_eigenvalues` histograms) are
  evaluated pointwise**, not vectorized inside the reused module — fine at the default
  grid sizes, slower if `n_grid`/`bins` is pushed very high.

Spec: vault `Modules/viz.md` (private). Tests: `tests/viz/tests.py` (oracles:
`scipy.stats`, `statsmodels`, `scipy.cluster.hierarchy`, `lifelines`, numpy/scipy linear
algebra, and direct brute-force formulas), `tests/viz/e2e.py` (API sweep), and
`tests/viz/backend_mpl.py` (real-matplotlib rendering checks, run only by the
`viz-matplotlib` CI job — every other test in this module passes with no matplotlib
installed at all).
