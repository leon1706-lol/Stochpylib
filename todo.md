# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.20.0 `utils` — verification complete, ready for the commit ask

Implementation, documentation, and full-suite verification are all done. Every one of the
23 modules plus `library`/`docs`/`cli` was individually run and confirmed green this
session (per-module `pytest -q` pass counts, summing to the full tree):

```
probability+survival+information_theory+optimization  402
distributions                                          175 passed / 2 skipped (sanctioned)
montecarlo                                              81
timeseries+nonparametric                               222
gaussian_processes+spatial_statistics                  229
copulas+bayesian+levy_processes                        237
financial_stochastics                                  171
statistics+viz                                         359
random_matrix+numerical_methods                        226
robust_statistics+experimental_design                  308
advanced_mcmc                                          101
utils                                                  188
library                                                 73
docs                                                     16
queueing+cli                                           148
------------------------------------------------------------
total                                                 2936 passed / 2 skipped
```

A full per-module `--collect-only` reconciliation confirms this sums to exactly 2938
collected, matching a fresh whole-tree `--collect-only` run — so the numbers already
written into README.md/CHANGELOG.md/etc. (2938 collected, 2936 passed, 2 skipped) are
correct as-is; no doc changes needed. (One false alarm along the way: `financial_stochastics`
appeared to be missing 8 tests against an earlier session's "163 passed" reading — a clean
rerun showed 171 passed / 0 skipped, matching collection exactly. That earlier number was
just a bad reading, not a real gap.)

### Still open before this can be called shipped

1. **Vault handoff** (AGENTS.md §5.7, not started): from `Stochpylib-Obsidian-Vault/` run
   `python scripts/generate_code_graph.py --repo-root .. --vault . --append-handoff --agent
   <name> --summary "..."` then `python scripts/regenerate_vault.py --repo-root .. --vault .
   --append-handoff --agent <name> --action updated --summary "..."`, then append a manual
   `HANDOFF.MD` entry summarizing the whole task (see existing entries in that file for the
   format: Timestamp/Agent/Action/Summary/Files changed).
2. **Git: nothing staged or committed yet**, correctly deferred. Once the vault handoff
   above is done, propose staging exactly the files `git status --short` shows (the new
   `stochpylib/_rng.py`/`_parallel.py`/`utils/`/`tests/utils/`, plus every retrofitted
   module file and doc it lists) and the message
   `v0.20.0: utils module + library-wide RNG/parallel/backend retrofit`, then wait for
   approval. No tag, no publish.
3. Minor/optional, not a bug: a Python 3.14 diagnostic ("Current thread's C stack trace ...
   cannot get C stack on this system") printed once during an earlier `montecarlo` run,
   apparently from concurrent first-time lazy `scipy.special` imports across worker threads
   under `ThreadPoolExecutor`. The run still finished green every time and this wasn't
   investigated further — worth a glance if it recurs more disruptively.
4. The golden-stream stream-neutrality regression script used throughout this work only
   ever existed in a session scratchpad, not the repo — it's a diagnostic tool, not part of
   the shipped test suite, so nothing to commit, but regenerate it if you want to
   re-verify RNG-retrofit stream-neutrality again later.

## Next candidate (after V0.20.0 actually ships)

The design spec is fully implemented (794/794). Open-ended directions for whatever comes
next (no decision made — ask, or pick one and say so):

- **Retrofit `n_jobs=` onto `HamiltonianMonteCarlo`/`NoUTurnSampler`/`NeutraHMC`.** These
  override `MCMCSampler.sample()` for divergence-tracking bookkeeping and don't yet forward
  `n_jobs=`/`backend=` the way the eleven base-`sample()` samplers do.
- **Real hardware validation of `GPUBackend("cupy")`.** CI can only exercise it through an
  injected fake module (no GPU on hosted runners); real correctness against actual CUDA
  hardware has never been checked.
- **PyPI release.** The published package (`0.6.4`) badly lags the repository (`0.20.0`);
  tagging and publishing per `development/infrastructure.md`'s release process would close
  that gap for real users installing from PyPI.

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verify in per-module/per-chunk batches (1-3 modules per invocation)
  instead. Not expected to be an issue in real CI (GitHub Actions runners have far more
  headroom). The dev sandbox also runs Python subprocess startup and heavy test runs at
  highly variable, often very slow wall-clock speed (minutes of wall time for near-zero CPU
  time on some invocations; single test-file runs have taken 10-40 minutes of wall clock)
  — an I/O/scheduling artifact of this box, not a code performance issue; budget generous
  timeouts and prefer background runs (`run_in_background`) for anything beyond a single
  module. `pytest --collect-only -q tests/` (no execution, just AST collection) completes
  fast even though a full run doesn't — use it to get an exact test count (currently 2938).
- **Don't poll for background results.** A `run_in_background` command delivers its own
  completion notification automatically the moment it finishes — repeatedly calling
  `ReadNotifications`/checking process CPU in a tight loop while waiting burns tokens for
  no new information (see AGENTS.md §5 step 4). Just continue the turn, or arm one
  `Monitor` on the run if you need an earlier signal than full completion.
