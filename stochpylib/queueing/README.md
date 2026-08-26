# stochpylib.queueing

Queueing theory and networks: closed-form single-queue results from M/M/1
through priority queues, birth-death formulas (Erlang B/C, Engset), product-form
networks (Jackson/closed/BCMP), and a discrete-event simulation engine to
cross-check them all.

**Status:** implemented & tested (29/29 spec names).

## Files

- `single_queues.py` — M/M/1 (closed-form), M/M/c (Erlang-C via birth-death),
  M/M/inf (no-wait limit), M/D/1 (Pollaczek-Khinchine), M/G/1 (P-K with
  second-moment input), GI/G/1 (Kingman heavy-traffic approximation),
  MG1PriorityQueue (non-preemptive two-class).
- `birth_death.py` — general birth-death steady-state solver, Erlang B/C
  formulas (iteratively stable), Engset formula.
- `networks.py` — JacksonNetwork (open, traffic equations via linear solve),
  OpenNetwork alias, ClosedNetwork/GordonNewell mean-value analysis, BCMP
  theorem types 1 and 3, ProductFormNetwork base class.
- `simulation.py` — `DiscreteEventSim` event-calendar engine with warmup
  filtering; `SimStats` collecting wait/sojourn/service times and time-averaged
  populations; `QueueSimulation` facade comparing analytical vs simulated.
- `analysis.py` — LittleLaw solver (solve for any missing variable),
  `traffic_intensity`, `mean_waiting_time`, `mean_queue_length`,
  `server_utilization`, `WaitingTimeDistribution` (exact M/M/1 and M/M/c CDF).

## Conventions

- All analytical models return a shared immutable `QueueResult` with fields
  `L`, `Lq`, `W`, `Wq`, `rho` and model-specific extras.
- Native numpy/scipy only; simulation seeds via `random_state=`.

## Known limitations

- GI/G/1 uses the Kingman approximation (accuracy degrades near rho = 1).
- BCMP supports only type-1 and type-3 station semantics.
- The simulation engine is single-station FIFO multi-server — no routing,
  blocking, or priority preemption.

Spec: vault `Modules/queueing.md` (private). Tests:
`tests/queueing/tests.py`.
