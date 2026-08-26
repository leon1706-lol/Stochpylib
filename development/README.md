# development/

Development-process documentation for stochpylib, kept separate from the root
`README.md` so that file stays short and scannable. Read every file here before
working on the library — they hold the structure and plan context.

| Document | Contents |
|---|---|
| [`Development.md`](Development.md) | Package layout decisions and workflow notes (why tests live outside the package, naming history, the wrap-up workflow) |
| [`architecture.md`](architecture.md) | The full system architecture: design thesis, module map, the common distribution contract, cross-cutting conventions, and per-module design notes |
| [`infrastructure.md`](infrastructure.md) | Build/packaging/CI runbook: local setup, the `spl` CLI, GitHub Actions workflows, the PyPI Trusted-Publisher release pipeline |
| [`project_structure.md`](project_structure.md) | The annotated directory tree: what every folder and top-level file is for |
| [`CHANGELOG.md`](CHANGELOG.md) | Detailed, append-only, per-phase build history: what was built, when, and why |
| [`Probleme.md`](Probleme.md) | Audit log of bugs and infrastructure issues, each with a severity rating (1–10) and a status from the legend at the top (fixed / partial / closed). Add an entry for every bug you find and fix |
| [`Implementation-Checklist.md`](Implementation-Checklist.md) | Every planned module/submodule/public name as checkboxes — the single-glance progress tracker (currently 317/794). Update it whenever you implement |

Module status snapshot: `probability`, `distributions`, `montecarlo`,
`timeseries`, `gaussian_processes`, `copulas`, `survival`, `queueing` and
`information_theory` are complete and tested (317/794 spec names across nine
modules); all other modules are still planned — see the checklist for the
authoritative per-name state.

Wrap-up procedure for any task: see the vault's `Essential-Tasks.md`
(`../Stochpylib-Obsidian-Vault/Essential-Tasks.md`).
