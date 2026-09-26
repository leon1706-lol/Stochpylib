# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.18.0 — done

`spatial_statistics` shipped end to end (32/32 spec names, 721/794 overall, twenty-one
modules): variograms and fitting, simple/ordinary/universal/co-/indicator/disjunctive
kriging, covariance-driven random fields, point processes with Ripley's K/pair
correlation, and spatial autocorrelation tests, plus `SARModel`/`CARModel` lattice extras
(closing the vault's CAR/SAR gap) and its `smoke (spatial_statistics)` CI job. Also: every
remaining `scipy.stats` import removed from library code (9 files, Probleme.md #114) with
a permanent AST guard test, and `spl show` now disambiguates public names exported by more
than one module (`GaussianRandomField`: `levy_processes` + `spatial_statistics`). Five real
bugs surfaced and were fixed on the way (Probleme.md #115-#118).

## Next candidate

Two modules remain: `viz` (35), `utils` (38). Each needs an explicit decision before it can
start:

- `viz`: a plotting-backend decision — matplotlib as a dev-only oracle/example dependency,
  or a from-scratch SVG/ASCII renderer to keep the zero-runtime-dep policy. It would also
  render `experimental_design`'s data-only `InteractionPlot`/`NormalPlot`, and
  `spatial_statistics.SpatialCovariance`/`Semivariogram`/`SpatialFunction` are natural
  `plot_variogram()`/`plot_...` targets.
- `utils.performance` (`GPUBackend`, `JIT_compile()`): optional torch/cupy/numba deps and a
  lazy-import design pass first.

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom). The dev
  sandbox also runs Python subprocess startup and heavy test runs at highly variable,
  often very slow wall-clock speed (minutes of wall time for near-zero CPU time on some
  invocations) — an I/O/scheduling artifact of this box, not a code performance issue;
  budget generous timeouts and prefer background runs for anything beyond a single module.
