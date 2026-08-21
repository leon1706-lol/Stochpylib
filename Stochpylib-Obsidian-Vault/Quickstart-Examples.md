# Quickstart Examples (target API)

These are the usage examples from the design spec component — they describe the **intended**
public API surface, not code that currently runs. Use them as the contract to implement against:
if an implementation can't satisfy one of these calls, either the implementation or this file is
wrong, and the mismatch should be resolved deliberately (update whichever is incorrect) rather than
left to drift.

## Installation

```bash
pip install stochpylib
```

## Time Series — GARCH Volatility

```python
from stochpylib.timeseries import GARCH, ARIMA
import numpy as np

returns = np.random.randn(1000) * 0.01

arima = ARIMA(p=1, d=0, q=1).fit(returns)
residuals = arima.residuals()

garch = GARCH(p=1, q=1).fit(residuals)
volatility = garch.conditional_volatility()
forecast = garch.forecast(horizon=10)

from stochpylib.timeseries import KalmanFilter
kf = KalmanFilter(F=..., H=..., Q=..., R=...)
kf.fit(observations)
smoothed = kf.smooth()
```

## Gaussian Processes

```python
from stochpylib.gaussian_processes import GPRegression
from stochpylib.gaussian_processes.kernels import RBFKernel, MaternKernel

X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(50)

kernel = RBFKernel(length_scale=1.0) + MaternKernel(nu=2.5)
gp = GPRegression(kernel=kernel, noise=0.01)
gp.fit(X_train, y_train)

X_test = np.linspace(0, 12, 200).reshape(-1, 1)
mu, sigma = gp.predict(X_test, return_std=True)
```

## Copulas & Dependence

```python
from stochpylib.copulas import GaussianCopula, VineCopula, CopulaFit

data = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 1000)
copula = GaussianCopula()
copula.fit(data)
samples = copula.sample(500)
tau = copula.kendall_tau()

vc = VineCopula(type="DVine")
vc.fit(data_5d)
```

## Survival Analysis

```python
from stochpylib.survival import KaplanMeier, CoxProportionalHazards

durations = [5, 8, 12, 20, 33, 45]
events    = [1, 1,  0,  1,  1,  0]

km = KaplanMeier()
km.fit(durations, events)
km.plot()
print(km.median_survival_time())

cox = CoxProportionalHazards()
cox.fit(X_covariates, durations, events)
print(cox.hazard_ratios())
```

## Hawkes Process

```python
from stochpylib.levy_processes import HawkesProcess, MultivariateHawkes

hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
events = hp.simulate(T=100)
print(f"Number of events: {len(events)}")

hp_fit = HawkesProcess()
hp_fit.fit(observed_events)
print(hp_fit.branching_ratio())
hp_fit.ks_residuals()
```

## Financial Stochastics

```python
from stochpylib.financial_stochastics import BlackScholes, HestonModel, ValueAtRisk

bs = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)
print(bs.call_price())      # 10.45
print(bs.Delta, bs.Gamma)

heston = HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04,
                      xi=0.3, rho=-0.7, r=0.05)
paths = heston.simulate(T=1, N=252, n_paths=10000)

var = ValueAtRisk(confidence=0.99)
var.historical(returns_data)
```

## Advanced MCMC — NUTS

```python
from stochpylib.advanced_mcmc import NoUTurnSampler, SequentialMonteCarlo
from stochpylib.advanced_mcmc.diagnostics import Rhat, ESS

def log_posterior(theta):
    return -0.5 * np.sum((y - X @ theta)**2) / sigma2

nuts = NoUTurnSampler(log_posterior, n_samples=2000,
                       n_warmup=500, target_accept=0.8)
nuts.sample(theta_init=np.zeros(p))
chains = nuts.get_chains()

print(f"R-hat: {Rhat(chains)}")
print(f"ESS:   {ESS(chains)}")
```

## Random Matrix Theory

```python
from stochpylib.random_matrix import GOE, WishartMatrix, MarchenkoPastur

goe = GOE(n=500)
eigenvalues = goe.eigenvalues()
goe.plot_semicircle()

W = WishartMatrix(p=200, n=1000)
eigs = W.eigenvalues()
mp = MarchenkoPastur(gamma=0.2)
mp.plot_vs_empirical(eigs)
```

## Variance Reduction

```python
from stochpylib.montecarlo.variance_reduction import (
    AntitheticVariates, ControlVariates, SobolSequence
)

av = AntitheticVariates(n_simulations=100_000)
price = av.price_european_call(S=100, K=100, T=1, r=0.05, sigma=0.2)

sobol = SobolSequence(dim=5)
X = sobol.generate(n=10000)
```

---
[[Module-Map]] · [[ARCHITECTURE]]
