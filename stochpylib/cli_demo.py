"""Runnable mini-examples behind ``spl demo <module>``.

One fast, deterministic, dependency-light demo per implemented module: fixed
seeds, small data, a few seconds each, ASCII-safe output. Each demo prints real
computed numbers against the live installation — the quickest way for a new
user to see a module actually work. Run via ``spl demo`` (bare: lists the
available demos) or programmatically through :func:`run_demo`.
"""

import numpy as np

__all__ = ["DEMO_MODULES", "run_demo", "list_demos"]


def _demo_probability():
    from stochpylib.probability import bayes_theorem, total_probability

    print("Bayes' theorem - disease screening (1% prevalence, 99% sensitivity,")
    print("5% false-positive rate):")
    p_pos = total_probability((0.99, 0.01), (0.05, 0.99))
    p_dis = bayes_theorem(0.01, 0.99, p_pos)
    print(f"  P(positive)          = {p_pos:.4f}")
    print(f"  P(disease | positive)= {p_dis:.4f}   <- only ~17%, base rates matter")


def _demo_distributions():
    from stochpylib.distributions import Normal, Weibull

    print("Distributions - one interface, closed forms + MLE fit + KS test:")
    d = Normal(0.0, 1.0)
    print(f"  N(0,1): pdf(0)={d.pdf(0.0):.4f} cdf(1.96)={d.cdf(1.96):.4f} "
          f"ppf(0.975)={d.ppf(0.975):.4f}")
    data = Weibull(2.0, 10.0).rvs(300, random_state=1)
    fitted = Weibull.fit(data)
    stat, p = fitted.ks_test(data)
    print(f"  Weibull.fit on n=300 drawn from Weibull(k=2, theta=10):")
    print(f"    recovered shape={fitted.shape:.3f} scale={fitted.scale:.3f} "
          f"(KS p={p:.3f})")


def _demo_montecarlo():
    from stochpylib.distributions import Normal
    from stochpylib.montecarlo import AntitheticVariates, SobolSequence

    print("Monte Carlo - quasi-random points and variance reduction:")
    pts = SobolSequence(dim=3).generate(4_096)
    print(f"  SobolSequence(dim=3).generate(4096) -> {pts.shape}, "
          f"dim-0 mean={pts[:, 0].mean():.5f} (exact block mean 0.5)")
    price = AntitheticVariates(n_simulations=50_000).price_european_call(
        S=100, K=100, T=1, r=0.05, sigma=0.2)
    sq = 0.2  # sigma * sqrt(T), T = 1
    d1 = (np.log(100 / 100) + (0.05 + sq * sq / 2)) / sq
    d2 = d1 - sq
    nd = lambda x: Normal(0, 1).cdf(x)
    bs = 100 * nd(d1) - 100 * np.exp(-0.05) * nd(d2)
    print(f"  European call (S=K=100, T=1, r=5%, sigma=20%):")
    print(f"    MC   = {price.estimate:.4f} +- {price.std_error:.4f}")
    print(f"    BS   = {bs:.4f}  (Black-Scholes oracle)")


def _demo_timeseries():
    from stochpylib.timeseries import AR

    print("Time series - AR(1) fit and forecast on synthetic data:")
    rng = np.random.default_rng(3)
    y = np.zeros(300)
    for t in range(1, 300):
        y[t] = 0.7 * y[t - 1] + 0.3 + 0.2 * rng.standard_normal()
    model = AR(1).fit(y)
    fc = np.asarray(model.forecast(3).mean)
    phi = float(np.atleast_1d(model.ar_coefs_)[0])
    print(f"  true phi=0.70 -> fitted phi={phi:.3f} "
          f"(intercept={float(model.intercept_):.3f})")
    print(f"  3-step forecast: {np.round(fc, 3)}")


def _demo_gaussian_processes():
    from stochpylib.gaussian_processes import GPRegression, RBFKernel

    print("Gaussian processes - regression with uncertainty:")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 25)[:, None]
    y = np.sin(x[:, 0]) + 0.05 * rng.standard_normal(25)
    gp = GPRegression(kernel=RBFKernel(length_scale=1.0), noise=0.01).fit(x, y)
    mu, sd = gp.predict(np.array([[2.5]]), return_std=True)
    print(f"  trained on 25 noisy sin(x) points (RBF kernel)")
    print(f"  predict(2.5): mean={float(mu[0]):.4f} (true sin(2.5)={np.sin(2.5):.4f}), "
          f"std={float(sd[0]):.4f}")


def _demo_copulas():
    from stochpylib.copulas import CopulaFit, kendall_tau
    from stochpylib.distributions import Student_t

    print("Copulas - dependence modeling with AIC family selection:")
    data = Student_t(4.0).rvs((600, 2), random_state=2)
    fit = CopulaFit().fit(data)
    print(f"  600 bivariate heavy-tailed returns")
    print(f"  best family by AIC: {type(fit.best_).__name__}, "
          f"Kendall tau = {float(kendall_tau(data)):.3f}")
    sim = fit.best_.sample(1_000)
    print(f"  sampled {sim.shape[0]} dependent pairs from the fitted copula")


def _demo_survival():
    from stochpylib.survival import KaplanMeier

    print("Survival - Kaplan-Meier on censored data:")
    rng = np.random.default_rng(4)
    durations = rng.exponential(5.0, 80)
    events = rng.random(80) < 0.8
    km = KaplanMeier().fit(durations, events)
    s = np.atleast_1d(km.predict([2.0, 5.0]))
    true_s = np.exp(-np.array([2.0, 5.0]) / 5.0)
    print(f"  80 subjects, {int(events.sum())} events, {int((~events).sum())} censored")
    print(f"  S(2)={s[0]:.3f} (true {true_s[0]:.3f}), "
          f"S(5)={s[1]:.3f} (true {true_s[1]:.3f})")


def _demo_queueing():
    from stochpylib.queueing import MM1Queue

    print("Queueing - M/M/1 closed form (lambda=0.8, mu=1.0):")
    q = MM1Queue().fit(arrival_rate=0.8, service_rate=1.0)
    print(f"  rho={q.rho:.2f} L={q.L:.3f} Lq={q.Lq:.3f} W={q.W:.3f} Wq={q.Wq:.3f}")
    print("  (system holds 4 customers on average; 4.0 time units waiting)")


def _demo_information_theory():
    from stochpylib.information_theory import Entropy, HuffmanCode

    print("Information theory - entropy and optimal coding:")
    probs = [0.5, 0.25, 0.125, 0.125]
    h = Entropy(base=2).compute(probs)
    hc = HuffmanCode().fit(probs=probs)
    print(f"  source {probs}")
    print(f"  entropy H = {h:.4f} bits/symbol, Huffman avg length = "
          f"{hc.average_length_:.4f} (optimal: {hc.is_optimal_})")
    print(f"  code table: {hc.code_table_}")


DEMOS = {
    "probability": _demo_probability,
    "distributions": _demo_distributions,
    "montecarlo": _demo_montecarlo,
    "timeseries": _demo_timeseries,
    "gaussian_processes": _demo_gaussian_processes,
    "copulas": _demo_copulas,
    "survival": _demo_survival,
    "queueing": _demo_queueing,
    "information_theory": _demo_information_theory,
}


def list_demos():
    """Print the available demos (the keys of ``DEMOS``)."""
    print("Available demos (spl demo <module>):")
    for name in DEMOS:
        print(f"  {name}")


def run_demo(module=None):
    """Run one module demo (or list them when ``module`` is None).

    Returns a process exit code: 0 on success, 1 when the module is unknown.
    """
    if module is None:
        list_demos()
        return 0
    fn = DEMOS.get(module)
    if fn is None:
        import difflib
        import stochpylib

        print(f"unknown demo module: {module!r}")
        known = list(stochpylib.__all__)
        close = difflib.get_close_matches(module, known, n=3)
        if close:
            print(f"did you mean: {', '.join(close)}?")
        print("run bare 'spl demo' to list the available demos")
        return 1
    fn()
    return 0
