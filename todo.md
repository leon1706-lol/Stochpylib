# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.19.0 — done

`viz` shipped end to end (35/35 spec names, 756/794 overall, twenty-two modules): a native,
zero-dependency SVG renderer (scene graph in `_figure.py`, rendering in `_svg.py`, nice-number
ticks, categorical + viridis/RdBu colormaps) with matplotlib as an optional, lazily-imported
backend (`_mpl.py`, confined there and guarded by an AST test) for `Figure.to_matplotlib()`/
raster-PDF output. All 35 functions reuse the rest of the library's own numbers rather than
re-deriving them — distributions' ppf-based grids, timeseries' spectral functions,
advanced_mcmc's Rhat/ESS, statsmodels-matching ACF/PACF, spatial_statistics' variograms,
random_matrix's limit laws, lifelines-matching Kaplan-Meier, and more. `experimental_design`'s
`InteractionPlot`/`NormalPlot` gained a `to_figure()` hook. A new `viz-matplotlib` CI job
(ubuntu + windows) installs matplotlib and runs the real-backend suite
(`tests/viz/backend_mpl.py`, excluded from the main pytest collection); every other viz test
passes with no matplotlib installed at all. One real pre-existing bug was found and fixed on
the way: `timeseries.CWTTransform`'s default scale range always crashed (the reflect-padding
was sized to the series length, not to the wavelet's own support, so any scale beyond ~n/20
overran the valid-convolution length) — Probleme.md #120.

## Next candidate

One module remains: `utils` (38). It needs an explicit decision before it can start:

- `utils.performance` (`GPUBackend`, `JIT_compile()`) and `utils.compat` (`torch_interface()`,
  `jax_interface()`, `pandas_interface()`): optional torch/cupy/numba/jax/pandas deps behind
  a lazy-import design, mirroring the pattern `viz`'s optional matplotlib backend just
  established (confine each optional import to one file, guard with an AST test, add a CI job
  that installs the real backend rather than skipping it).

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom). The dev
  sandbox also runs Python subprocess startup and heavy test runs at highly variable,
  often very slow wall-clock speed (minutes of wall time for near-zero CPU time on some
  invocations) — an I/O/scheduling artifact of this box, not a code performance issue;
  budget generous timeouts and prefer background runs for anything beyond a single module.
  `pytest --collect-only -q tests/` (no execution, just AST collection) completes fine on
  the whole tree even though a full run doesn't — use it to get an exact test count.
