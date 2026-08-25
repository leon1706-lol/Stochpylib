# stochpylib.queueing — queueing theory & networks

**Status: implemented and tested** (29/29 spec names).

Layout:

- `single_queues.py` — M/M/1 (closed-form), M/M/c (Erlang-C via birth-death),
  M/M/∞, M/D/1 (Pollaczek-Khinchine), M/G/1 (P-K with second-moment input),
  GI/G/1 (Kingman heavy-traffic approximation), MG1PriorityQueue
  (non-preemptive two-class). All return a shared `QueueResult`.
- `birth_death.py` — general birth-death steady-state solver,
  Erlang B/C formulas (iteratively stable), Engset formula.
- `networks.py` — JacksonNetwork (open, traffic equations via linear solve),
  OpenNetwork alias, ClosedNetwork/GordonNewell mean-value analysis,
  BCMP theorem types 1 and 3, ProductFormNetwork base class.
- `simulation.py` — DiscreteEventSim event-calendar engine with warmup
  filtering; SimStats collecting wait/sojourn/service times and time-averaged
  populations; QueueSimulation facade comparing analytical vs simulated.
- `analysis.py` — LittleLaw solver (solve for any missing variable),
  traffic_intensity, mean_waiting_time, mean_queue_length,
  server_utilization, WaitingTimeDistribution (exact MM1/MMC CDF).

Conventions: all models return a shared immutable `QueueResult` with fields
L, Lq, W, Wq, rho and model-specific extras. Native numpy/scipy only.

Known limitations: GI/G/1 uses Kingman approximation (accuracy degrades near
rho=1); BCMP supports only type-1 and type-3 stations; simulation is
single-station FIFO multi-server (no routing or blocking).

Spec: vault `Modules/queueing.md` (private). Tests: `tests/queueing/tests.py`.
