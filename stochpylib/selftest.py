"""Embedded self-check suite for stochpylib.

Ships inside the installed package so ``spl --test`` can verify any installation
(pip-installed copies included) without pytest or a source checkout. This is a fast
smoke test — the full test suite lives in ``tests/`` at the repository root and runs
via ``pytest tests/ -v``.

Run programmatically via :func:`run` (returns the number of failing checks).
"""

import numpy as np

import stochpylib
from stochpylib.distributions import (
    AlphaStable,
    Bernoulli,
    Beta,
    Binomial,
    Cauchy,
    Chi2,
    Dirichlet,
    Exponential,
    Gamma,
    Geometric,
    LevyDistribution,
    LogNormal,
    MultivariateNormal,
    Normal,
    Poisson,
    Rayleigh,
    StableDistribution,
    Student_t,
    Uniform,
    Weibull,
)

_TOL = 1e-6


class _SelfTest:
    def __init__(self):
        self.failures = []
        self.count = 0

    def check(self, name, cond):
        self.count += 1
        if not cond:
            self.failures.append(name)
            print(f"  FAIL {name}")
        return bool(cond)


def _univariate_checks(st, name, d, discrete=False, moments=True):
    lo, hi = d.support()
    x = float(d.ppf(0.4)) if np.isfinite(lo) or np.isfinite(hi) else 0.3
    p = float(np.atleast_1d(np.asarray(d.pdf(x), dtype=float))[0])
    st.check(f"{name}: pdf>=0", np.isfinite(p) and p >= 0.0)
    c = float(d.cdf(x))
    st.check(f"{name}: cdf in [0,1]", 0.0 <= c <= 1.0)
    q = 0.5
    xp = float(d.ppf(q))
    cp = float(d.cdf(xp))
    if discrete:
        st.check(f"{name}: ppf semantics", cp >= q)
    else:
        st.check(f"{name}: cdf(ppf(q))==q", abs(cp - q) < 1e-4)
    smp = np.asarray(d.rvs(50, random_state=0), dtype=float)
    st.check(f"{name}: rvs finite", bool(np.all(np.isfinite(smp))))
    if moments:
        m, v = d.mean(), d.var()
        if np.isfinite(m) and np.isfinite(v):
            st.check(f"{name}: var>0", v >= 0.0)


def run(verbose=False):
    """Run all self-checks. Returns the number of failing checks (0 == success)."""
    st = _SelfTest()
    if verbose:
        print(f"stochpylib selftest - version {stochpylib.__version__}")

    # package sanity
    st.check("package: version string", isinstance(stochpylib.__version__, str))
    st.check("package: probability import", hasattr(getattr(stochpylib, "probability"), "bayes_theorem"))
    bt = stochpylib.probability.bayes_theorem(0.01, 0.99, 0.0197)
    st.check("probability: bayes sanity", abs(bt - 0.502539) < 0.001)

    # univariate distributions (one small instance each; closed-form spot values where cheap)
    n = Normal(0.0, 1.0)
    st.check("Normal: pdf(0)", abs(n.pdf(0.0) - 0.3989422804014327) < _TOL)
    st.check("Normal: mean/var", abs(n.mean()) < _TOL and abs(n.var() - 1.0) < _TOL)
    _univariate_checks(st, "Normal", n)

    e = Exponential(2.0)
    st.check("Exponential: mean", abs(e.mean() - 0.5) < _TOL)
    _univariate_checks(st, "Exponential", e)

    b = Bernoulli(0.3)
    st.check("Bernoulli: pmf(1)", abs(float(b.pmf(1)) - 0.3) < _TOL)
    _univariate_checks(st, "Bernoulli", b, discrete=True)

    bi = Binomial(10, 0.4)
    st.check("Binomial: mean", abs(bi.mean() - 4.0) < _TOL)
    _univariate_checks(st, "Binomial", bi, discrete=True)

    po = Poisson(3.0)
    st.check("Poisson: pmf sums", abs(float(np.sum(po.pmf(np.arange(0, 60))))) > 1.0 - 1e-8)
    _univariate_checks(st, "Poisson", po, discrete=True)

    g = Geometric(0.25)
    st.check("Geometric: mean", abs(g.mean() - 4.0) < _TOL)
    _univariate_checks(st, "Geometric", g, discrete=True)

    u = Uniform(0.0, 1.0)
    st.check("Uniform: pdf height", abs(u.pdf(0.5) - 1.0) < _TOL)
    _univariate_checks(st, "Uniform", u)

    be = Beta(2.0, 3.0)
    st.check("Beta: mean", abs(be.mean() - 0.4) < _TOL)
    _univariate_checks(st, "Beta", be)

    ga = Gamma(2.0, 1.0)
    st.check("Gamma: mean", abs(ga.mean() - 2.0) < _TOL)
    _univariate_checks(st, "Gamma", ga)

    c2 = Chi2(3)
    st.check("Chi2: mean", abs(c2.mean() - 3.0) < _TOL)
    _univariate_checks(st, "Chi2", c2)

    t5 = Student_t(5)
    st.check("Student_t: var", abs(t5.var() - 5.0 / 3.0) < _TOL)
    _univariate_checks(st, "Student_t", t5)

    ca = Cauchy(0.0, 1.0)
    st.check("Cauchy: median", abs(ca.ppf(0.5)) < _TOL)
    _univariate_checks(st, "Cauchy", ca, moments=False)

    we = Weibull(2.0, 1.0)
    st.check("Weibull: cdf(1)", abs(we.cdf(1.0) - (1 - np.exp(-1.0))) < _TOL)
    _univariate_checks(st, "Weibull", we)

    ln = LogNormal(0.0, 0.5)
    st.check("LogNormal: mean", abs(ln.mean() - np.exp(0.125)) < _TOL)
    _univariate_checks(st, "LogNormal", ln)

    ra = Rayleigh(1.0)
    st.check("Rayleigh: mean", abs(ra.mean() - np.sqrt(np.pi / 2)) < _TOL)
    _univariate_checks(st, "Rayleigh", ra)

    # stable family: exact special-case delegation + fast CML sampling
    s2 = StableDistribution(2.0, 0.0, 0.0, 1.0)
    st.check(
        "Stable alpha=2 == Gaussian",
        abs(float(s2.pdf(0.7)) - float(Normal(0.0, np.sqrt(2)).pdf(0.7))) < _TOL,
    )
    sc = StableDistribution(1.0, 0.0, 0.0, 1.0)
    st.check("Stable alpha=1,beta=0 == Cauchy", abs(float(sc.pdf(1.3)) - float(Cauchy(0, 1).pdf(1.3))) < _TOL)
    sa = AlphaStable(1.5, 0.0, 1.0)
    xs = np.asarray(sa.rvs(20000, random_state=1))
    st.check("AlphaStable CML rvs: symmetric median~loc", abs(float(np.median(xs)) - 0.0) < 0.05)
    lv = LevyDistribution(0.0, 1.0)
    st.check("Levy: support", lv.support()[0] == 0.0)
    _univariate_checks(st, "Levy", lv, moments=False)

    if verbose:
        status = "OK" if not st.failures else f"FAILED ({len(st.failures)})"
        print(f"selftest: {st.count} checks, {status}")

    return len(st.failures)


if __name__ == "__main__":
    raise SystemExit(1 if run(verbose=True) else 0)
