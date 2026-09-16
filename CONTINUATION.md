# CONTINUATION — levy_processes Module (V0.7.0)

**Status: COMPLETE.** This file tracked an interrupted session; a follow-up
session picked it up and finished every item below end-to-end. Kept here as a
record of what was found and fixed — safe to delete once reviewed. See
`development/CHANGELOG.md` (Phase 23) and `development/Probleme.md` (#41-50)
for the permanent record.

**Final version:** V0.7.0 (614 collected / 612 passed / 2 skipped)
**Spec names:** 33 of 794 (317 + 33 = 350 / 794)

---

## What was finished in the follow-up session

1. **The two CRITICAL bugs left open were fixed** — but the actual root
   causes were different from what this file originally guessed:
   - `TemperingSubordinator`/`CGMYProcess` mean bug: not a shortfall-rate
     numerical issue — the truncated-jump quantile grid was linear where the
     Lévy density is singular, so a naive Riemann-sum CDF overweighted the
     near-floor region ~2x. Fixed with a log-spaced grid + cumulative
     trapezoid quadrature. Mean is now within 0.3% (was ~50% low).
   - `KouJumpDiffusion.call_price` bug: not the risk-neutral compensator —
     `carr_madan_call()` used `k = log(K/S0)` (relative log-moneyness) when
     the characteristic function it integrates already embeds `log(S0)` in
     its own drift term and needs the *absolute* log-strike `k = log(K)`.
     Reproduced with a bare GBM CF against the library's own Black-Scholes
     (99.05 vs the exact 10.45), independent of anything Kou-specific.

2. **Nine more bugs were found while writing the test suite** (not
   suspected by the original session): `CoxProcess.simulate` crashing on
   numpy >= 2.x (`np.trapz` removed); `StableSubordinator`/
   `RandomMeasure(kind="stable")` not matching their documented Laplace
   transform (uncancelled S1-parameterization constant); `Runge_Kutta_SDE`
   actually being strong order ~0.5 (a stochastic-Heun scheme mislabeled
   "order 1.0" — replaced with the derivative-free Milstein/Platen scheme);
   `StochasticTaylor` actually being strong order ~1.0, not the documented
   1.5 (wrong multiple-stochastic-integral formulas — rewritten to match
   Kloeden-Platen exactly); `StrongApproximation` sharing one already-advanced
   RNG between the "exact" and "approximate" solvers (so it never measured a
   real strong error at all — this is *why* the RK/Taylor bugs were invisible
   until this was fixed first); `WeakApproximation`'s three-point increment
   distribution using 1/3-1/3-1/3 weights instead of 1/6-2/3-1/6 (doubled
   `E[dW**2]`, producing a non-vanishing `E[X_T]` bias); `HawkesProcess.ks_residuals()`
   always raising when called with no arguments right after `.fit()` (fitted
   events were never stored); and `TemperingSubordinator.truncation_mass()`'s
   docstring describing the opposite of what it computes (documentation-only).
   Full detail: `development/Probleme.md` #41-51.

3. **`tests/levy_processes/tests.py` written** (39 tests): subordinators,
   Lévy-Khintchine/stable-process CF consistency, jump-diffusion pricing vs
   Black-Scholes + Monte Carlo, Hawkes/Cox/renewal/branching/semi-Markov/
   random-field/random-measure checks, and SDE strong/weak convergence-order
   studies — several are direct regressions for the bugs above.

4. **Every Essential-Tasks.md wrap-up item completed:** `selftest.py`
   extended 139 -> 145 checks; `stochpylib/levy_processes/README.md` written;
   `development/CHANGELOG.md`, `Probleme.md`, `Implementation-Checklist.md`
   (33/33, progress line 350/794), `architecture.md`, `infrastructure.md`
   updated; root `README.md` (badges, status table, Known Limitations,
   architecture diagrams, roadmap, CLI reference counts), `stochpylib/README.md`,
   `tests/README.md`, `CONTRIBUTING.md`, `development/Development.md`,
   `development/README.md`, `development/project_structure.md` all synced;
   `pyproject.toml` bumped to 0.7.0 (was still 0.6.4 — the prior session had
   only bumped `stochpylib/__init__.py`); `tests/docs/tests.py` and
   `tests/library/tests.py` updated to include `levy_processes` in their
   consistency/wiring checks; `spl demo levy_processes` added to
   `cli_demo.py` (and its `DEMO_MODULES` export bug fixed along the way).

5. **Not done (out of scope for this phase, left to the user):** git
   commit/push (explicitly requested to be done manually), and the private
   Obsidian vault handoff scripts (`Stochpylib-Obsidian-Vault/scripts/`) —
   the vault is gitignored/private and its regeneration doesn't affect
   anything that gets committed, so it was skipped to keep this phase's
   scope to the committable repository. Run it manually if you want the
   vault's code graph refreshed:
   ```bash
   cd Stochpylib-Obsidian-Vault
   python scripts/generate_code_graph.py --repo-root .. --vault . --append-handoff --agent "claude" --summary "implemented levy_processes"
   python scripts/regenerate_vault.py --repo-root .. --vault . --append-handoff --agent "claude" --action implemented --summary "levy_processes module"
   ```
