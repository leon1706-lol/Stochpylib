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


def _demo_levy_processes():
    from stochpylib.levy_processes import KouJumpDiffusion, TemperingSubordinator
    from stochpylib.levy_processes.jump_diffusion import _black_scholes_call

    print("Levy processes - jump-diffusion option pricing and a tempered-stable clock:")
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    kou = KouJumpDiffusion(mu=r, sigma=0.2, jump_rate=1.0, eta_up=10.0,
                           eta_down=10.0, p_up=0.4)
    price = kou.call_price(S0, K, T, r)
    mc = kou.call_price_mc(S0, K, T, r, n_paths=100_000, random_state=0)
    print(f"  Kou double-exponential jump-diffusion call (S=K=100, T=1, r=5%):")
    print(f"    Carr-Madan  = {price:.4f}")
    print(f"    Monte Carlo = {mc.estimate:.4f} +- {mc.std_error:.4f}")
    ts = TemperingSubordinator(C=1.0, lam=5.0, alpha=0.5)
    path = ts.sample([0.25, 0.5, 0.75, 1.0], random_state=1)
    print(f"  TemperingSubordinator path at t=0.25,0.5,0.75,1.0: {np.round(path, 4)}")
    print(f"    analytic mean rate = {ts.mean_rate():.4f} per unit time")


def _demo_financial_stochastics():
    from stochpylib.financial_stochastics import BlackScholes, HestonModel, ValueAtRisk

    print("Financial stochastics - Black-Scholes, Heston, and historical VaR/ES:")
    bs = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)
    print(f"  BlackScholes(S=100,K=105,T=1,r=5%,sigma=20%) call = {bs.call_price():.4f}"
          f"  Delta={bs.Delta:.4f}  Gamma={bs.Gamma:.4f}")

    heston = HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05)
    cm_price = heston.call_price(100, 1)
    mc = heston.call_price_mc(100, 1, n_paths=100_000, N=100, scheme="qe", random_state=0)
    print(f"  Heston call (Carr-Madan) = {cm_price:.4f}  vs QE Monte Carlo = {mc.estimate:.4f} +- {mc.std_error:.4f}")

    rng = np.random.default_rng(2)
    returns = rng.normal(0.0004, 0.012, 1000)
    var = ValueAtRisk(confidence=0.99)
    res = var.historical(returns)
    print(f"  Historical 99% VaR on 1000 synthetic daily returns = {res.estimate:.4f}"
          f"  ES = {res.expected_shortfall:.4f}")


def _demo_statistics():
    from stochpylib.statistics import ANOVA, PCA, describe, linear_regression, t_test

    print("Statistics - describe, t-test, OLS regression, and PCA:")
    rng = np.random.default_rng(3)
    x = rng.normal(5.0, 1.5, 200)
    d = describe(x)
    print(f"  describe(n=200 samples): mean={d.mean:.4f}  std={d.std:.4f}  skew={d.skewness:.4f}")

    y = rng.normal(5.3, 1.5, 180)
    t = t_test(x, y, equal_var=False)
    print(f"  Welch t-test: t={t.statistic:.4f}  p={t.pvalue:.4f}  diff={t.estimate:.4f}")

    X = rng.normal(size=(300, 2))
    yl = 1.0 + X @ np.array([2.0, -1.5]) + rng.normal(0, 0.5, 300)
    ols = linear_regression(X, yl)
    print(f"  OLS: coef={np.round(ols.coef_, 4)}  R2={ols.r2_:.4f}")

    pca = PCA(X, n_components=2)
    print(f"  PCA explained variance ratio = {np.round(pca.explained_variance_ratio_, 4)}")


def _demo_random_matrix():
    from stochpylib.random_matrix import GOE, EigenvalueSpacing, MarchenkoPastur, WishartMatrix

    print("Random matrix theory - semicircle, Marchenko-Pastur, and level repulsion:")
    goe = GOE(300)
    eig = goe.eigenvalues(random_state=0)
    res = goe.limit_law().compare(goe.normalize(eig))
    print(f"  GOE(300): eigenvalues/sqrt(n) vs Wigner semicircle -> KS D={res.statistic:.4f} "
          f"(p={res.pvalue:.3f})")
    W = WishartMatrix(p=100, n=400)
    mp = MarchenkoPastur(gamma=0.25)
    e = W.normalized_eigenvalues(random_state=1)
    print(f"  Wishart(p=100, n=400): eigenvalues/n in [{e.min():.3f}, {e.max():.3f}] vs "
          f"Marchenko-Pastur support [{mp.lam_minus:.3f}, {mp.lam_plus:.3f}], KS p={mp.compare(e).pvalue:.3f}")
    r = EigenvalueSpacing(eig).mean_ratio()
    print(f"  GOE mean adjacent-gap ratio <r> = {r:.4f}  (GOE 0.5307, Poisson 0.3863) -> "
          f"{EigenvalueSpacing(eig).classify()}")


def _demo_advanced_mcmc():
    import numpy as np

    from stochpylib.advanced_mcmc import ESS, NoUTurnSampler, Rhat, SequentialMonteCarlo, SliceSampling

    print("Advanced MCMC - NUTS on a correlated Gaussian, slice sampling, SMC evidence:")
    mu = np.array([1.0, -1.0])
    rho = 0.6
    prec = np.linalg.inv([[1.0, rho], [rho, 1.0]])

    def log_prob(theta):
        d = theta - mu
        return -0.5 * d @ prec @ d

    def grad_log_prob(theta):
        return -prec @ (theta - mu)

    nuts = NoUTurnSampler(log_prob, grad_log_prob, n_samples=1500, n_warmup=500, n_chains=2, target_accept=0.8)
    nuts.sample(theta_init=np.zeros(2), random_state=0)
    chains = nuts.get_chains()
    print(f"  NUTS(2 chains, correlated Gaussian): R-hat={Rhat(chains).max():.4f}, "
          f"ESS={ESS(chains).min():.0f}, mean accept-stat={nuts.acceptance_rate_:.3f}")

    def log_prob_gamma(theta):
        x = theta[0]
        return -np.inf if x <= 0 else 2.0 * np.log(x) - x / 2.0  # Gamma(3, scale=2), mean 6

    sl = SliceSampling(log_prob_gamma, n_samples=2000, n_warmup=300, width=3.0)
    sl.sample(np.array([6.0]), random_state=1)
    print(f"  Slice sampler on Gamma(3, scale=2): sample mean = {sl.get_samples().mean():.3f} (true 6.000)")

    y = np.array([1.0])

    def log_prior(theta):
        return -0.5 * np.sum(theta ** 2)

    def log_lik(theta):
        return -0.5 * np.sum(((theta - y) / 0.5) ** 2)

    def prior_sampler(n, rng):
        return rng.standard_normal((n, 1))

    smc = SequentialMonteCarlo(log_prior, log_lik, prior_sampler, n_particles=1000, n_mcmc=3)
    smc.sample(random_state=2)
    logZ_true = -0.5 * np.log(2 * np.pi * 1.25) - 0.5 * y[0] ** 2 / 1.25
    print(f"  SMC log evidence = {smc.log_evidence_:.4f} (closed form {logZ_true:.4f})")


def _demo_numerical_methods():
    from stochpylib.financial_stochastics import BlackScholes
    from stochpylib.numerical_methods import Brent, DormandPrince, FiniteDifference, GaussLegendre, MatrixExponential

    print("Numerical methods - Gauss quadrature, Dormand-Prince, Brent, PDE pricing:")
    gl = GaussLegendre(5)
    r = gl.integrate(np.sin, 0, np.pi)
    print(f"  5-point Gauss-Legendre integral of sin over [0,pi] = {r.value:.6f} (exact 2.0)")

    def f(t, y):
        return np.array([y[1], -y[0]])

    dp = DormandPrince(rtol=1e-8)
    sol = dp.solve(f, (0, 2 * np.pi), [1.0, 0.0])
    energy0 = 1.0 ** 2 + 0.0 ** 2
    energyT = sol.y[-1, 0] ** 2 + sol.y[-1, 1] ** 2
    print(f"  Dormand-Prince harmonic oscillator: energy drift after one period = "
          f"{abs(energyT - energy0):.2e}")

    S, K, T, r_rate, true_sigma = 100.0, 100.0, 1.0, 0.05, 0.25
    target = BlackScholes(S=S, K=K, T=T, r=r_rate, sigma=true_sigma).call_price()
    root = Brent(lambda sig: BlackScholes(S=S, K=K, T=T, r=r_rate, sigma=sig).call_price() - target,
                0.01, 2.0)
    print(f"  Brent implied volatility recovered = {root.root:.4f} (true 0.2500)")

    fd = FiniteDifference()
    pde_price = fd.black_scholes(K, T, r_rate, true_sigma, kind="call").price(S)
    bs_price = BlackScholes(S=S, K=K, T=T, r=r_rate, sigma=true_sigma).call_price()
    print(f"  Crank-Nicolson PDE call price = {pde_price:.4f} vs Black-Scholes = {bs_price:.4f}")

    Q = np.array([[-2.0, 1.0, 1.0], [1.0, -3.0, 2.0], [0.5, 0.5, -1.0]])
    P = MatrixExponential(Q).at(1.0)
    print(f"  3-state CTMC transition matrix expm(Q) row sums = {np.round(P.sum(axis=1), 6)}")


def _demo_bayesian():
    from stochpylib.bayesian import (
        BayesianLinear, BayesianNetwork, WAIC, bayes_factor, likelihood, posterior, prior,
    )
    from stochpylib.distributions import Beta

    print("Bayesian inference - conjugate posterior, linear regression, model selection:")
    rng = np.random.default_rng(7)
    flips = rng.binomial(1, 0.7, 10)
    post = posterior(prior(Beta(2, 2)), likelihood("bernoulli", data=flips))
    lo, hi = post.credible_interval(0.95)
    print(f"  coin flips (7/10 heads): posterior mean={float(post.mean()[0]):.3f}, "
          f"95% CI=({lo:.3f}, {hi:.3f})")

    X = rng.standard_normal((60, 1))
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.5, 60)
    model = BayesianLinear().fit(X, y)
    beta_s, sigma2_s = model.sample(1000, random_state=0)
    w = WAIC(model.pointwise_log_lik((beta_s, sigma2_s)))
    print(f"  linear regression: coef={np.round(model.coef_, 3)}, WAIC={w.value:.2f}")

    null_model = BayesianLinear().fit(X * 0.0, y)
    bf = bayes_factor(model, null_model)
    print(f"  Bayes factor (slope model vs intercept-only) = {bf.value:.3g} "
          f"({bf.extras['jeffreys']})")

    bn = BayesianNetwork()
    for name in ("Cloudy", "Sprinkler", "Rain", "WetGrass"):
        bn.add_node(name, [0, 1])
    bn.add_edge("Cloudy", "Sprinkler")
    bn.add_edge("Cloudy", "Rain")
    bn.add_edge("Sprinkler", "WetGrass")
    bn.add_edge("Rain", "WetGrass")
    bn.set_cpt("Cloudy", [0.5, 0.5])
    bn.set_cpt("Sprinkler", [[0.5, 0.5], [0.9, 0.1]])
    bn.set_cpt("Rain", [[0.8, 0.2], [0.2, 0.8]])
    wg = np.zeros((2, 2, 2))
    wg[0, 0] = [1.0, 0.0]; wg[0, 1] = [0.1, 0.9]
    wg[1, 0] = [0.1, 0.9]; wg[1, 1] = [0.01, 0.99]
    bn.set_cpt("WetGrass", wg)
    res = bn.query(["Rain"], {"WetGrass": 1})
    print(f"  sprinkler network: P(Rain=1 | WetGrass=1) = {res[1]:.4f}")


def _demo_robust_statistics():
    from stochpylib.robust_statistics import (
        HodgesLehmann, MCD, Median, MedianAbsoluteDeviation, MMRegression,
        TheilSenRegression, BlockBootstrap,
    )
    from stochpylib.statistics import linear_regression

    print("Robust statistics - location/scale, regression, and covariance under")
    print("contamination:")
    rng = np.random.default_rng(11)
    readings = np.concatenate([rng.normal(23.5, 0.8, 85), rng.uniform(-10, 60, 15)])
    print(f"  15% contaminated sensor readings: mean={np.mean(readings):.2f}  "
          f"median={Median().fit(readings).estimate_:.2f}  "
          f"Hodges-Lehmann={HodgesLehmann().fit(readings).estimate_:.2f}")
    print(f"  MAD scale={MedianAbsoluteDeviation().fit(readings).estimate_:.2f} "
          f"vs naive std={np.std(readings, ddof=1):.2f}")

    t = rng.uniform(0, 10, 100)
    y = 1.0 + 2.0 * t + rng.normal(0, 0.3, 100)
    y[:30] += 20.0  # 30% gross outliers
    ols = linear_regression(t, y)
    ts = TheilSenRegression().fit(t, y)
    mm = MMRegression(random_state=0).fit(t, y)
    print(f"  30% outlier regression (true slope=2.0): OLS={ols.coef_[1]:.3f}  "
          f"Theil-Sen={ts.coef_[1]:.3f}  MM={mm.coef_[1]:.3f}")

    mu = np.array([23.5, 45.2, 1013.0])
    Sigma = np.array([[1.0, 0.3, -0.2], [0.3, 2.0, 0.1], [-0.2, 0.1, 3.0]])
    cal = mu + rng.standard_normal((300, 3)) @ np.linalg.cholesky(Sigma).T
    bad = rng.choice(300, 45, replace=False)
    cal[bad] += rng.normal(0, 1, (45, 3)) * 10
    mcd = MCD(random_state=0).fit(cal)
    print(f"  MCD on a 15% contaminated 3-D calibration sample: "
          f"{mcd.outliers().sum()} of 45 planted outliers flagged")

    bb = BlockBootstrap(np.mean, random_state=0).fit(readings)
    print(f"  moving-block bootstrap SE of the mean = {bb.std_error_:.3f}")


def _demo_nonparametric():
    from stochpylib.nonparametric import (
        DistanceCorrelation, IsotonicRegression, KernelDensityEstimate,
        KruskalWallis, LocalPolynomialReg,
    )

    print("Nonparametric methods - density estimation, rank tests, dependence,")
    print("and smoothing, none of it assuming a parametric family:")
    rng = np.random.default_rng(21)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(6, 1.5, 100)])
    kde = KernelDensityEstimate(bandwidth="silverman").fit(x)
    print(f"  bimodal sample (n={len(x)}): KDE pdf(0)={kde.pdf(0.0):.3f}  "
          f"pdf(6)={kde.pdf(6.0):.3f}  bandwidth={kde.bandwidth_:.3f}")

    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0, 1, 30)
    g3 = rng.normal(2.5, 1, 30)
    kw = KruskalWallis().fit(g1, g2, g3)
    print(f"  Kruskal-Wallis on 3 groups (one shifted +2.5): "
          f"H={kw.statistic_:.3f}  p={kw.pvalue_:.4g}")

    a = rng.uniform(-2, 2, 150)
    b = a ** 2 + rng.normal(0, 0.3, 150)
    dcor = DistanceCorrelation(n_resamples=300, random_state=1).fit(a, b)
    print(f"  y = x^2 + noise: distance correlation={dcor.estimate_:.3f}  "
          f"p={dcor.pvalue_:.4g} (Pearson would miss this)")

    t = np.sort(rng.uniform(0, 10, 100))
    z = np.sin(t) + rng.normal(0, 0.2, 100)
    lp = LocalPolynomialReg(bandwidth=0.8).fit(t, z)
    print(f"  local-linear smoother on sin(t)+noise: R^2={lp.score(t, z):.3f}  "
          f"effective df={lp.df_:.1f}")

    yi = np.sort(rng.uniform(0, 1, 40)) + rng.normal(0, 0.3, 40)
    iso = IsotonicRegression().fit(np.arange(40), yi)
    print(f"  isotonic regression: {int(np.sum(np.diff(iso.fitted_) < -1e-10))} "
          f"monotonicity violations in the fit (expect 0)")


def _demo_optimization():
    from stochpylib.optimization import (
        AugmentedLagrangian, BFGS, CMA_ES, LevenbergMarquardt,
        ParticleSwarmOptimization,
    )

    print("Optimization - the same minimize(fun, x0) interface across gradient,")
    print("derivative-free and constrained methods, each reporting an OptimizeResult:")
    rng = np.random.default_rng(23)

    rosen = lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    rosen_der = lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                                    200 * (x[1] - x[0] ** 2)])
    bfgs = BFGS().minimize(rosen, [-1.2, 1.0], grad=rosen_der)
    print(f"  Rosenbrock from (-1.2, 1.0): BFGS x=({bfgs.x_[0]:.6f}, {bfgs.x_[1]:.6f})  "
          f"f={bfgs.fun_:.2e}  {bfgs.result_.nit} iterations, {bfgs.result_.nfev} evals")

    rastrigin = lambda x: 10 * len(x) + float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))
    pso = ParticleSwarmOptimization(bounds=(-5.12, 5.12), n_particles=40, n_iter=300,
                                    random_state=1).minimize(rastrigin, np.full(5, 3.0))
    print(f"  Rastrigin-5 (10^5 local minima): particle swarm f={pso.fun_:.3e} "
          f"vs f={rastrigin(np.full(5, 3.0)):.1f} at the start")

    cma = CMA_ES(sigma0=2.0, n_iter=400, random_state=2).minimize(rastrigin, np.full(5, 3.0))
    cond = np.linalg.cond(cma.result_.extras["C"])
    print(f"  same problem, CMA-ES: f={cma.fun_:.3e}  learned covariance condition "
          f"number={cond:.2f}")

    t = np.linspace(0, 4, 60)
    y = 2.5 * np.exp(-0.7 * t) + rng.normal(0, 0.02, len(t))
    lm = LevenbergMarquardt().minimize(lambda p: p[0] * np.exp(-p[1] * t) - y, [1.0, 1.0])
    print(f"  noisy exponential decay: Levenberg-Marquardt amplitude={lm.x_[0]:.4f} "
          f"rate={lm.x_[1]:.4f} (true 2.5 / 0.7)")

    con = AugmentedLagrangian(
        constraints=[{"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}]
    ).minimize(lambda x: float(x[0] ** 2 + x[1] ** 2), [2.0, -1.0])
    print(f"  min x^2+y^2 s.t. x+y=1: augmented Lagrangian x=({con.x_[0]:.6f}, "
          f"{con.x_[1]:.6f})  constraint violation={con.result_.extras['violation']:.1e}")


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
    "levy_processes": _demo_levy_processes,
    "financial_stochastics": _demo_financial_stochastics,
    "statistics": _demo_statistics,
    "random_matrix": _demo_random_matrix,
    "advanced_mcmc": _demo_advanced_mcmc,
    "numerical_methods": _demo_numerical_methods,
    "bayesian": _demo_bayesian,
    "robust_statistics": _demo_robust_statistics,
    "nonparametric": _demo_nonparametric,
    "optimization": _demo_optimization,
}
DEMO_MODULES = tuple(DEMOS)


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
