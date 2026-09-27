# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.20.4


- cut down on code comments by deleting unecesary comments
- cuting somewhat important comments to one or two short sentences
- only leaving important commants
- only touch code files no md documentation files 
- add to agent md the convention of how to write clean code commands 


# Next candidate (after V0.20.0 actually ships)

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
  fast even though a full run doesn't — use it to get an exact test count (currently 2939).
- **Don't poll for background results.** A `run_in_background` command delivers its own
  completion notification automatically the moment it finishes — repeatedly calling
  `ReadNotifications`/checking process CPU in a tight loop while waiting burns tokens for
  no new information (see AGENTS.md §5 step 4). Just continue the turn, or arm one
  `Monitor` on the run if you need an earlier signal than full completion.
