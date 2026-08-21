# Module Reference

Planning-stage reference for every top-level `stochpylib` package, generated from the design spec (the React preview component) rather than from real source — there is no implementation yet in this repo. Treat each file below as the target shape for that package: its submodules and the public names it should expose.

Once real source exists, regenerate per-module *code* notes with `scripts/generate_code_graph.py` and keep these files as the design intent to diff against.

| Module | Score /10 | Submodules | Items | Notes |
|---|---|---|---|---|
| [[probability]] | 10 | 3 | 21 | Perfect coverage: Bayes, combinatorics, independence, conditional probability |
| [[distributions]] | 10 | 5 | 60 | 50+ distributions, all moments, MGF, CF, fitting — industry complete |
| [[timeseries]] | 10 | 9 | 61 | AR to GARCH family, state-space, HMM, wavelets, spectral — nothing missing |
| [[gaussian_processes]] | 10 | 5 | 36 | Full kernel zoo, sparse GP, regression & classification — research-grade |
| [[copulas]] | 9 | 5 | 26 | All major families + vine copulas; empirical copula is a strong add |
| [[survival]] | 9 | 6 | 28 | KM, Nelson-Aalen, Cox PH, AFT — covers clinical + reliability |
| [[queueing]] | 9 | 5 | 29 | M/M/1 to Jackson networks; LittleLaw is a must-have |
| [[information_theory]] | 9 | 5 | 31 | Transfer entropy & Jensen-Shannon push beyond standard textbooks |
| [[random_matrix]] | 8 | 4 | 23 | GOE/GUE, Marchenko-Pastur — niche but correct scope |
| [[levy_processes]] | 10 | 5 | 33 | Levy, Hawkes, Cox, jump-diffusion, semi-Markov — exhaustive |
| [[spatial_statistics]] | 8 | 5 | 32 | Kriging + variogram + point processes — solid but could add CAR/SAR |
| [[robust_statistics]] | 8 | 5 | 28 | Huber, RANSAC, Theil-Sen — essential for real-world data |
| [[nonparametric]] | 9 | 5 | 31 | KDE, bootstrap, rank tests — complete nonparametric toolkit |
| [[financial_stochastics]] | 10 | 7 | 50 | Black-Scholes to SABR, Greeks, VaR, credit risk — quant-ready |
| [[advanced_mcmc]] | 10 | 6 | 35 | NUTS, HMC, SMC, RJMCMC, particle MCMC — state of the art |
| [[montecarlo]] | 9 | 4 | 25 | Sobol, Halton, LHS, antithetic — professional simulation toolkit |
| [[bayesian]] | 9 | 4 | 25 | Prior/posterior/model selection — needs more variational inference |
| [[optimization]] | 8 | 5 | 32 | Adam, PSO, SA, genetic — good mix of classical and heuristic |
| [[numerical_methods]] | 8 | 6 | 38 | Quadrature, FEM, FDM — solid backbone for simulation |
| [[experimental_design]] | 8 | 5 | 29 | LHS, factorial, response surface — covers DOE fundamentals |
| [[statistics]] | 9 | 5 | 48 | z/t/chi2/ANOVA + Kruskal-Wallis + permutation tests |
| [[viz]] | 9 | 5 | 35 | PDF/CDF/process/posterior/ACF — modern statistical plotting |
| [[utils]] | 9 | 6 | 38 | GPU backend + parallel simulation + reproducibility = production ready |

**Totals:** 23 modules · 120 submodules · 794+ public names.
