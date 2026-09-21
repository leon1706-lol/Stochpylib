# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.15.0 — done

`nonparametric` implemented end to end (31/31 spec names), tests + `module-smoke` CI job
added, full Essential-Tasks.md wrap-up completed. See `development/CHANGELOG.md` Phase 32
for the full breakdown. 628/794 public names, eighteen modules implemented.

## Next candidate

Same size-first heuristic as before (smallest remaining module next): the five remaining
modules are `spatial_statistics` (32), `optimization` (32), `experimental_design` (29),
`viz` (35), `utils` (38). `experimental_design` is the smallest and has no third-party-dep
caveat. `utils.performance` (`GPUBackend`, `JIT_compile()`) still implies optional
torch/cupy/numba deps needing a lazy-import design pass before it's picked up. `viz`
implies a plotting backend decision (matplotlib as a dev-only oracle/example dependency,
or a from-scratch SVG/ASCII renderer to keep the zero-runtime-dep policy — needs an
explicit decision before starting, unlike the other four).

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).
