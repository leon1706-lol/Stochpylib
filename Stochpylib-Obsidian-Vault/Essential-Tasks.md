# Essential Tasks — Standard Wrap-Up Checklist

Reference this file as context when implementing any piece of stochpylib, instead of re-typing the
checklist by hand each time. This checklist assumes you've just implemented (or changed) a
class/function inside one of the [Module-Map.md](Module-Map.md) modules.

## 1. Check the design spec before and after writing code

Before implementing, read the relevant `Modules/<name>.md` file and
[Quickstart-Examples.md](Quickstart-Examples.md) for the target API shape (constructor args,
method names, return types). After implementing:
- If your implementation matches the spec, you're done with this step.
- If you deliberately deviated (better API, spec was wrong/ambiguous), update the corresponding
  `Modules/<name>.md` entry and, if it's a flagship example, [Quickstart-Examples.md](Quickstart-Examples.md)
  so the vault doesn't silently drift from reality.
- Check the cross-cutting conventions in [ARCHITECTURE.md](ARCHITECTURE.md) (distribution method
  set, fit/predict symmetry, kernel composability) — new code should follow them unless there's a
  specific reason not to.

## 2. Manual debugging session

Don't rely on unit tests alone. In a scratch script, hand-construct a realistic use case (e.g. fit
a GARCH model to synthetic returns and forecast, or fit a Kaplan-Meier curve to a small dataset)
and run it end-to-end against the real implementation, not mocks. Fix anything it surfaces before
moving on.

## 3. Add/extend tests

- **If you just created a new module, you must create its `tests.py` as part of the same task —
  not as follow-up work.** A module isn't done until `tests/<module_name>/tests.py` exists and
  covers it.
- Tests live outside the package, in a top-level `tests/<module_name>/tests.py` per module —
  e.g. `tests/probability/tests.py`, `tests/timeseries/tests.py` — not colocated inside the
  package and not split per-submodule.
- If the module already has a `tests.py` (you extended an existing module), add to that file
  rather than creating a second one.
- For distributions specifically, verify the common interface contract from
  [ARCHITECTURE.md](ARCHITECTURE.md) holds (`.pdf()`, `.cdf()`, `.rvs()`, `.fit()`, etc.) —
  that consistency is load-bearing for the rest of the library.

## 4. Update documentation

1. Mark the module's status in `Modules/<name>.md` (`planned` → `implemented`) and correct any
   API details that changed during implementation.
2. **Update [Implementation-Checklist.md](../development/Implementation-Checklist.md)** — check off every item
   you actually implemented and tested (don't check off a submodule until every item in it is
   done). This is the single-glance progress tracker; it must stay in sync with reality.
3. Update [Ratings.md](Ratings.md) only if implementation revealed the original design score was
   wrong (missing/extra scope) — not just because code now exists.
4. If this is the first module implemented, replace "Status: pre-implementation" language in
   [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) with accurate current status.
5. Update `../development/CHANGELOG.md` with a new entry describing what was built, and
   `../development/Probleme.md` with any bug you found and fixed (or left open) along the way —
   include a severity score (1–10) and a Fixed/Open status for each entry.

## 5. Run the full test suite

```bash
pytest tests/ -v
```

Must be green before moving on. Don't proceed to regenerating the vault graph on a red suite.

## 6. Regenerate the Obsidian vault's code notes

Once a module has real source, generate per-file/per-function code notes to supplement (not
replace) the hand-authored `Modules/` files:

```bash
python scripts/generate_code_graph.py --repo-root .. --vault . --append-handoff \
  --agent "<agent name>" --summary "<short summary of the change>"

python scripts/regenerate_vault.py --repo-root .. --vault . --append-handoff \
  --agent "<agent name>" --action updated --summary "<short summary>"
```

Run both from inside `Stochpylib-Obsidian-Vault/` — confirm with `pwd` first, since running from
`scripts/` writes a stray `HANDOFF.MD` one level too deep. Both scripts append their own
`HANDOFF.MD` entry automatically via `--append-handoff`.

## 7. Append a manual HANDOFF.MD entry

In addition to the scripts' auto-appended entries, add your own entry to `HANDOFF.MD` summarizing
the task as a whole (Timestamp/Agent/Action/Summary/Files changed) — the scripts only know about
the vault-regeneration step, not the actual implementation work.

## 8. Final sanity check

- Full test suite green.
- `git status --short` — confirm the diff only touches what you intended.
