# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.16.0 — done

`optimization` shipped end to end (32/32 spec names, 660/794 overall): gradient and
adaptive-step methods, quasi-Newton/trust-region/Levenberg-Marquardt, seven
metaheuristics, stochastic approximation, and five constrained solvers. The red
`test (3.10, *)` jobs from the V0.15.0 push were fixed first (Probleme.md #102), so the
module landed on a green baseline. Six further bugs found in the manual debug session are
recorded as Probleme.md #103-#108.

## Next candidate

Four modules remain: `spatial_statistics` (32), `experimental_design` (29), `viz` (35),
`utils` (38). Smallest-first puts `experimental_design` next, and it has no third-party-dep
caveat. Two still need an explicit decision before they can start:

- `viz` implies a plotting-backend decision — matplotlib as a dev-only oracle/example
  dependency, or a from-scratch SVG/ASCII renderer to keep the zero-runtime-dep policy.
- `utils.performance` (`GPUBackend`, `JIT_compile()`) implies optional torch/cupy/numba
  deps and needs a lazy-import design pass first.

`spatial_statistics` also needs a naming decision: its `GaussianRandomField` collides with
the one already shipped in `levy_processes`.

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).
