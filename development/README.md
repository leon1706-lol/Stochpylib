# development/ — build history & process docs

Read every file here before working on the library — they hold the structure and plan context:

- `Development.md` — package layout decisions and workflow notes (why tests live outside the
  package, naming history).
- `CHANGELOG.md` — append-only log, one entry per meaningful chunk of work.
- `Probleme.md` — bug audit log: every entry as **Problem → Fix → Verification** with a
  severity (1–10) and a status from the legend at the top (🟢 fixed / 🟡 partial /
  🔴 closed). Add an entry for every bug you find and fix.
- `Implementation-Checklist.md` — every planned module/submodule/public name as checkboxes;
  the single-glance progress tracker (currently 229/794). Update it whenever you implement.
- Module status snapshot: `probability`, `distributions`, `montecarlo`, `timeseries`,
  `gaussian_processes` and `copulas` are complete and tested (229/794 spec names); all
  other modules are still planned — see the checklist for the authoritative per-name state.

Wrap-up procedure for any task: see the vault's `Essential-Tasks.md`.
