# stochpylib.queueing

*Queueing theory & networks*

**Design-completeness score:** 9/10 — M/M/1 to Jackson networks; LittleLaw is a must-have

Status: **planned** — not yet implemented.

## Submodules

### `queueing.single_queues`

- `MM1Queue`
- `MMCQueue`
- `MMInfinityQueue`
- `MD1Queue`
- `MG1Queue`
- `GI1Queue`
- `GIGQueue`
- `MG1PriorityQueue`

### `queueing.networks`

- `JacksonNetwork`
- `GordonNewell`
- `BCMP`
- `OpenNetwork`
- `ClosedNetwork`
- `ProductFormNetwork`

### `queueing.analysis`

- `LittleLaw()`
- `traffic_intensity()`
- `mean_waiting_time()`
- `mean_queue_length()`
- `server_utilization()`
- `WaitingTimeDistribution`
- `SojournTime`

### `queueing.birth_death`

- `BirthDeathQueue`
- `ErlangBFormula()`
- `ErlangCFormula()`
- `EngsetFormula()`

### `queueing.simulation`

- `QueueSimulation`
- `EventDrivenSim`
- `DiscreteEventSim`
- `SimStats`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
