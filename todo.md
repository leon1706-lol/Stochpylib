# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.14.0 — done

`robust_statistics` (28 names) shipped end to end: robust location/scale, high-breakdown
regression, robust covariance, resampling. 17/23 modules, 597/794 public names, version
0.14.0. Full test suite + e2e sweep + module-smoke CI job added; all docs synced.

## Next candidate

Same size-first heuristic as before (smallest remaining module next): `nonparametric` (31
names) and `spatial_statistics` (32) are the two smallest of the six remaining
(`nonparametric`, `spatial_statistics`, `optimization`, `experimental_design`, `viz`,
`utils`) — `nonparametric` edges it out. `utils.performance` (`GPUBackend`,
`JIT_compile()`) still implies optional third-party deps (torch/cupy/numba) needing a
lazy-import design pass before it's picked up, same caveat as before.

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).
