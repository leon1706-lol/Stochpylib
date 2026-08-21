# stochpylib.copulas

*Dependence modeling with copulas*

**Design-completeness score:** 9/10 — All major families + vine copulas; empirical copula is a strong add

Status: **planned** — not yet implemented.

## Submodules

### `copulas.elliptical`

- `GaussianCopula`
- `StudentTCopula`

### `copulas.archimedean`

- `ClaytonCopula`
- `FrankCopula`
- `GumbelCopula`
- `JoeCopula`
- `AliMikhailHaqCopula`
- `PlackettCopula`
- `BB1Copula`
- `BB7Copula`

### `copulas.empirical`

- `EmpiricalCopula`
- `CheckerboardCopula`
- `BetaCopula`

### `copulas.vine`

- `VineCopula`
- `CVine`
- `DVine`
- `RVine`
- `PairCopulaConstruction`
- `VineStructureSelect()`

### `copulas.methods`

- `CopulaFit()`
- `CopulaSample()`
- `kendall_tau()`
- `spearman_rho()`
- `tail_dependence()`
- `copula_density()`
- `conditional_copula()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
