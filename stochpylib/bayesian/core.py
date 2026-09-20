"""Priors, likelihoods, conjugate updating, posteriors and evidence.

Ten conjugate families (``ConjugateFamily``) cover the classic exponential-family cases
in closed form; ``posterior()``/``evidence()`` fall back to a 1-2 D numerical grid, a
Laplace approximation, variational inference, importance sampling, sequential Monte
Carlo, or MCMC (all delegating to :mod:`stochpylib.advanced_mcmc` /
:mod:`stochpylib.numerical_methods`) as the problem grows.
"""

import numpy as np
from scipy import special

from stochpylib.bayesian._common import _as_1d, _as_2d, _log_multibeta, _mvn_logpdf, _rng
from stochpylib.bayesian._result import Posterior

__all__ = ["prior", "likelihood", "posterior", "bayes_update", "conjugate_prior",
           "posterior_predictive", "evidence", "Prior", "Likelihood", "ConjugateFamily"]


def _prior_dim(dist):
    """Dimensionality of a library distribution/``_NormalInverseGamma`` used as a prior."""
    from stochpylib.bayesian._common import _NormalInverseGamma
    if isinstance(dist, _NormalInverseGamma):
        return 2  # (mu, sigma2)
    if hasattr(dist, "k"):  # MultivariateNormal, MultivariateT
        return int(dist.k)
    if hasattr(dist, "alpha") and np.ndim(dist.alpha) > 0:  # Dirichlet
        return len(dist.alpha)
    return 1


# --------------------------------------------------------------------------- Prior

class Prior:
    """A prior distribution over one or more parameters.

    ``dist``: one library ``Distribution``/``MultivariateDistribution``, a list/tuple of
    ``Distribution``s (independent product prior), or the string ``"flat"`` (improper:
    ``logpdf`` is identically 0, ``sample`` raises). Custom priors instead give
    ``logpdf(theta)`` (+ optional ``sampler(n, rng)`` and ``dim``).
    """

    def __init__(self, dist=None, *, logpdf=None, sampler=None, dim=None, name=None):
        self.name = name
        self.is_proper = True
        if dist is not None:
            if isinstance(dist, str):
                if dist != "flat":
                    raise ValueError("string dist must be 'flat'")
                self.dist = "flat"
                self.is_proper = False
                self.dim = dim if dim is not None else 1
            elif isinstance(dist, (list, tuple)):
                self.dist = list(dist)
                self.dim = len(self.dist)
            else:
                self.dist = dist
                self.dim = _prior_dim(dist)
            self._logpdf = None
            self._sampler = None
        elif logpdf is not None:
            self.dist = None
            self._logpdf = logpdf
            self._sampler = sampler
            if dim is None:
                raise ValueError("custom priors require dim")
            self.dim = int(dim)
        else:
            raise ValueError("give either dist or logpdf")

    def logpdf(self, theta):
        from stochpylib.bayesian._common import _NormalInverseGamma
        if self._logpdf is not None:
            return float(self._logpdf(theta))
        if self.dist == "flat":
            return 0.0
        if isinstance(self.dist, list):
            theta = np.atleast_1d(np.asarray(theta, dtype=float))
            total = 0.0
            for d, th in zip(self.dist, theta):
                p = np.clip(float(np.asarray(d.pdf(th))), 1e-300, None)
                total += np.log(p)
            return float(total)
        if isinstance(self.dist, _NormalInverseGamma):
            theta = np.atleast_1d(np.asarray(theta, dtype=float))
            return self.dist.logpdf(theta[0], theta[1])
        if self.dim == 1:
            p = np.clip(float(np.atleast_1d(np.asarray(self.dist.pdf(theta)))[0]), 1e-300, None)
            return float(np.log(p))
        p = np.clip(float(np.asarray(self.dist.pdf(np.asarray(theta, dtype=float)))), 1e-300, None)
        return float(np.log(p))

    def sample(self, n=1, random_state=None):
        from stochpylib.bayesian._common import _NormalInverseGamma
        rng = _rng(random_state)
        if self._sampler is not None:
            return np.atleast_2d(self._sampler(n, rng))
        if self.dist == "flat":
            raise ValueError("cannot sample from an improper flat prior")
        if isinstance(self.dist, list):
            cols = [np.atleast_1d(np.asarray(d.rvs(n, random_state=rng))) for d in self.dist]
            return np.column_stack(cols)
        if isinstance(self.dist, _NormalInverseGamma):
            return np.atleast_2d(self.dist.rvs(n, random_state=rng))
        out = np.atleast_2d(np.asarray(self.dist.rvs(n, random_state=rng)))
        if self.dim == 1 and out.shape[-1] != 1:
            out = out.reshape(-1, 1)
        return out

    def __repr__(self):
        return f"Prior(dim={self.dim}, proper={self.is_proper})"


def prior(dist=None, **kw):
    """Factory for :class:`Prior` (returns ``dist`` unchanged if already one)."""
    if isinstance(dist, Prior):
        return dist
    return Prior(dist, **kw)


# --------------------------------------------------------------------------- Likelihood

def _bincount_counts(labels, k):
    labels = np.asarray(labels, dtype=int)
    return np.bincount(labels, minlength=k).astype(float)


_FAMILIES = {
    "bernoulli": dict(dim=1),
    "binomial": dict(dim=1),
    "poisson": dict(dim=1),
    "exponential": dict(dim=1),
    "gamma": dict(dim=1),
    "normal": dict(dim=1),
    "normal_variance": dict(dim=1),
    "normal_unknown_var": dict(dim=2),
    "categorical": dict(dim=None),
    "mvnormal": dict(dim=None),
}


class Likelihood:
    """A likelihood for a batch of data under a named exponential family, or a custom
    ``loglik(theta, data)`` callable (+ optional ``pointwise(theta, data) -> (n,)``)."""

    def __init__(self, family=None, data=None, *, loglik=None, pointwise=None, **known):
        self.data = data
        self.known = known
        if family is None:
            if loglik is None:
                raise ValueError("give either family or loglik")
            self.family = None
            self._loglik = loglik
            self._pointwise = pointwise
            self.dim = known.get("dim")
        else:
            fam = family.lower()
            if fam not in _FAMILIES:
                raise ValueError(f"unknown family: {family!r} (known: {sorted(_FAMILIES)})")
            self.family = fam
            self._loglik = None
            self._pointwise = None
            self.dim = _FAMILIES[fam]["dim"]

    @property
    def n(self):
        if self.data is None:
            return 0
        if self.family == "binomial":
            return int(np.atleast_1d(np.asarray(self.data)).size)
        return len(_as_1d(self.data)) if np.ndim(self.data) <= 1 else np.asarray(self.data).shape[0]

    @property
    def conjugate_family(self):
        return self.family

    def pointwise(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        fam = self.family
        if fam is None:
            if self._pointwise is None:
                raise ValueError("pointwise log-likelihood not available for this custom likelihood")
            return np.asarray(self._pointwise(theta, self.data), dtype=float)
        data = self.data
        if fam == "bernoulli":
            p = np.clip(theta[0], 1e-12, 1.0 - 1e-12)
            x = _as_1d(data)
            return x * np.log(p) + (1 - x) * np.log(1 - p)
        if fam == "binomial":
            p = np.clip(theta[0], 1e-12, 1.0 - 1e-12)
            s = np.atleast_1d(np.asarray(data, dtype=float))
            n = np.atleast_1d(np.asarray(self.known["n"], dtype=float))
            n = np.broadcast_to(n, s.shape)
            return (special.gammaln(n + 1) - special.gammaln(s + 1) - special.gammaln(n - s + 1)
                    + s * np.log(p) + (n - s) * np.log(1 - p))
        if fam == "poisson":
            lam = max(theta[0], 1e-300)
            x = _as_1d(data)
            return -lam + x * np.log(lam) - special.gammaln(x + 1)
        if fam == "exponential":
            rate = max(theta[0], 1e-300)
            x = _as_1d(data)
            return np.log(rate) - rate * x
        if fam == "gamma":
            rate = max(theta[0], 1e-300)
            shape = self.known["shape"]
            x = _as_1d(data)
            return (shape * np.log(rate) - special.gammaln(shape) + (shape - 1) * np.log(x) - rate * x)
        if fam == "normal":
            mu = theta[0]
            sigma = self.known["sigma"]
            x = _as_1d(data)
            return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((x - mu) / sigma) ** 2
        if fam == "normal_variance":
            sigma2 = max(theta[0], 1e-300)
            mu = self.known["mu"]
            x = _as_1d(data)
            return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2
        if fam == "normal_unknown_var":
            mu, sigma2 = theta[0], max(theta[1], 1e-300)
            x = _as_1d(data)
            return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2
        if fam == "categorical":
            p = np.clip(theta, 1e-300, None)
            p = p / p.sum()
            if self.known.get("counts"):
                counts = np.asarray(data, dtype=float)
                return counts * np.log(p)
            labels = np.asarray(data, dtype=int)
            return np.log(p[labels])
        if fam == "mvnormal":
            cov = self.known["cov"]
            X = _as_2d(data)
            return np.array([_mvn_logpdf(row, theta, cov) for row in X])
        raise ValueError(fam)

    def loglik(self, theta):
        if self._loglik is not None:
            return float(self._loglik(theta, self.data))
        return float(np.sum(self.pointwise(theta)))

    def sample(self, theta, size=None, random_state=None):
        rng = _rng(random_state)
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        n = 1 if size is None else size
        fam = self.family
        if fam == "bernoulli":
            return rng.binomial(1, theta[0], size=n)
        if fam == "binomial":
            trials = int(self.known.get("n", 1))
            return rng.binomial(trials, theta[0], size=n)
        if fam == "poisson":
            return rng.poisson(theta[0], size=n)
        if fam == "exponential":
            return rng.exponential(1.0 / theta[0], size=n)
        if fam == "gamma":
            return rng.gamma(self.known["shape"], 1.0 / theta[0], size=n)
        if fam == "normal":
            return rng.normal(theta[0], self.known["sigma"], size=n)
        if fam == "normal_variance":
            return rng.normal(self.known["mu"], np.sqrt(theta[0]), size=n)
        if fam == "normal_unknown_var":
            return rng.normal(theta[0], np.sqrt(theta[1]), size=n)
        if fam == "categorical":
            p = np.clip(theta, 1e-300, None)
            p = p / p.sum()
            return rng.choice(len(p), size=n, p=p)
        if fam == "mvnormal":
            return rng.multivariate_normal(theta, self.known["cov"], size=n)
        raise ValueError("sample() is only available for the built-in families")

    def __repr__(self):
        fam = self.family or "custom"
        return f"Likelihood(family={fam!r}, n={self.n})"


def likelihood(family=None, data=None, **kw):
    """Factory for :class:`Likelihood`."""
    return Likelihood(family, data, **kw)


# --------------------------------------------------------------------- ConjugateFamily

class ConjugateFamily:
    """The conjugate-update rule for one likelihood family."""

    def __init__(self, family):
        fam = family.lower() if isinstance(family, str) else family.family
        if fam not in _CONJUGATE_RULES:
            raise ValueError(f"no conjugate family for {fam!r} "
                              f"(known: {sorted(_CONJUGATE_RULES)})")
        self.family = fam
        self.prior_ = None

    def default_prior(self):
        from stochpylib.distributions import Beta, Gamma, Normal, InvGamma, Dirichlet, MultivariateNormal
        from stochpylib.bayesian._common import _NormalInverseGamma
        fam = self.family
        if fam in ("bernoulli", "binomial"):
            return Beta(1.0, 1.0)
        if fam in ("poisson", "exponential", "gamma"):
            return Gamma(1e-2, 1e2)
        if fam == "normal":
            return Normal(0.0, 1e3)
        if fam == "normal_variance":
            return InvGamma(1e-2, 1e-2)
        if fam == "normal_unknown_var":
            return _NormalInverseGamma(0.0, 1e-3, 1e-2, 1e-2)
        if fam == "categorical":
            return None  # requires dimension; use make_prior(alpha=...)
        if fam == "mvnormal":
            return None  # requires dimension; use make_prior(mean=..., cov=...)
        raise ValueError(fam)

    def make_prior(self, **hyper):
        from stochpylib.distributions import Beta, Gamma, Normal, InvGamma, Dirichlet, MultivariateNormal
        from stochpylib.bayesian._common import _NormalInverseGamma
        fam = self.family
        if fam in ("bernoulli", "binomial"):
            d = Beta(hyper["a"], hyper["b"])
        elif fam in ("poisson", "exponential", "gamma"):
            d = Gamma(hyper["alpha"], 1.0 / hyper["beta"])
        elif fam == "normal":
            d = Normal(hyper["mu0"], hyper["tau0"])
        elif fam == "normal_variance":
            d = InvGamma(hyper["alpha"], hyper["beta"])
        elif fam == "normal_unknown_var":
            d = _NormalInverseGamma(hyper["mu"], hyper["kappa"], hyper["alpha"], hyper["beta"])
        elif fam == "categorical":
            d = Dirichlet(np.asarray(hyper["alpha"], dtype=float))
        elif fam == "mvnormal":
            d = MultivariateNormal(np.asarray(hyper["mean"], dtype=float), np.asarray(hyper["cov"], dtype=float))
        else:
            raise ValueError(fam)
        self.prior_ = d
        return d

    def hyperparameters(self, dist):
        fam = self.family
        if fam in ("bernoulli", "binomial"):
            return {"a": dist.a, "b": dist.b}
        if fam in ("poisson", "exponential", "gamma"):
            return {"alpha": dist.shape, "beta": 1.0 / dist.scale}
        if fam == "normal":
            return {"mu0": dist.mu, "tau0": dist.sigma}
        if fam == "normal_variance":
            return {"alpha": dist.shape, "beta": dist.scale}
        if fam == "normal_unknown_var":
            return {"mu": dist.mu, "kappa": dist.kappa, "alpha": dist.alpha, "beta": dist.beta}
        if fam == "categorical":
            return {"alpha": dist.alpha}
        if fam == "mvnormal":
            return {"mean": dist.mean_vec, "cov": dist.cov_matrix}
        raise ValueError(fam)

    def update(self, prior_dist, data, **known):
        return _CONJUGATE_RULES[self.family]["update"](prior_dist, data, known)

    def predictive(self, post_dist, **known):
        return _CONJUGATE_RULES[self.family]["predictive"](post_dist, known)

    def log_evidence(self, prior_dist, data, **known):
        return _CONJUGATE_RULES[self.family]["log_evidence"](prior_dist, data, known)

    def __repr__(self):
        return f"ConjugateFamily({self.family!r})"


def _rule_bernoulli_update(prior_dist, data, known):
    from stochpylib.distributions import Beta
    x = _as_1d(data)
    s, n = float(np.sum(x)), len(x)
    return Beta(prior_dist.a + s, prior_dist.b + n - s)


def _rule_bernoulli_predictive(post_dist, known):
    from stochpylib.distributions import Bernoulli
    return Bernoulli(post_dist.a / (post_dist.a + post_dist.b))


def _rule_bernoulli_evidence(prior_dist, data, known):
    x = _as_1d(data)
    s, n = float(np.sum(x)), len(x)
    a, b = prior_dist.a, prior_dist.b
    return float(special.betaln(a + s, b + n - s) - special.betaln(a, b))


def _rule_binomial_update(prior_dist, data, known):
    from stochpylib.distributions import Beta
    s = np.atleast_1d(np.asarray(data, dtype=float))
    n = np.broadcast_to(np.atleast_1d(np.asarray(known["n"], dtype=float)), s.shape)
    return Beta(prior_dist.a + s.sum(), prior_dist.b + (n - s).sum())


def _rule_binomial_predictive(post_dist, known):
    from stochpylib.distributions import BetaBinomial
    n_new = known.get("n")
    n_new = int(np.atleast_1d(n_new)[0]) if n_new is not None else 1
    return BetaBinomial(n_new, post_dist.a, post_dist.b)


def _rule_binomial_evidence(prior_dist, data, known):
    s = np.atleast_1d(np.asarray(data, dtype=float))
    n = np.broadcast_to(np.atleast_1d(np.asarray(known["n"], dtype=float)), s.shape)
    a, b = prior_dist.a, prior_dist.b
    log_binom = np.sum(special.gammaln(n + 1) - special.gammaln(s + 1) - special.gammaln(n - s + 1))
    return float(log_binom + special.betaln(a + s.sum(), b + (n - s).sum()) - special.betaln(a, b))


def _rule_poisson_update(prior_dist, data, known):
    from stochpylib.distributions import Gamma
    x = _as_1d(data)
    alpha, beta = prior_dist.shape, 1.0 / prior_dist.scale
    return Gamma(alpha + x.sum(), 1.0 / (beta + len(x)))


def _rule_poisson_predictive(post_dist, known):
    from stochpylib.distributions import NegBinomial
    alpha, beta = post_dist.shape, 1.0 / post_dist.scale
    return NegBinomial(alpha, beta / (beta + 1.0))


def _rule_poisson_evidence(prior_dist, data, known):
    x = _as_1d(data)
    n = len(x)
    alpha, beta = prior_dist.shape, 1.0 / prior_dist.scale
    a2, b2 = alpha + x.sum(), beta + n
    return float(-np.sum(special.gammaln(x + 1)) + special.gammaln(a2) - special.gammaln(alpha)
                 + alpha * np.log(beta) - a2 * np.log(b2))


def _rule_exponential_update(prior_dist, data, known):
    from stochpylib.distributions import Gamma
    x = _as_1d(data)
    alpha, beta = prior_dist.shape, 1.0 / prior_dist.scale
    return Gamma(alpha + len(x), 1.0 / (beta + x.sum()))


def _rule_exponential_predictive(post_dist, known):
    from stochpylib.distributions import GPareto
    alpha, beta = post_dist.shape, 1.0 / post_dist.scale
    return GPareto(loc=0.0, scale=beta / alpha, shape=1.0 / alpha)


def _rule_exponential_evidence(prior_dist, data, known):
    x = _as_1d(data)
    n = len(x)
    alpha, beta = prior_dist.shape, 1.0 / prior_dist.scale
    a2, b2 = alpha + n, beta + x.sum()
    return float(special.gammaln(a2) - special.gammaln(alpha) + alpha * np.log(beta) - a2 * np.log(b2))


def _rule_gamma_update(prior_dist, data, known):
    from stochpylib.distributions import Gamma
    x = _as_1d(data)
    k = known["shape"]
    alpha, beta = prior_dist.shape, 1.0 / prior_dist.scale
    return Gamma(alpha + len(x) * k, 1.0 / (beta + x.sum()))


def _rule_gamma_predictive(post_dist, known):
    return None  # compound gamma has no library class


def _rule_gamma_evidence(prior_dist, data, known):
    x = _as_1d(data)
    k = known["shape"]
    alpha, beta = prior_dist.shape, 1.0 / prior_dist.scale
    a2, b2 = alpha + len(x) * k, beta + x.sum()
    return float(np.sum((k - 1) * np.log(x) - special.gammaln(k))
                 + special.gammaln(a2) - special.gammaln(alpha) + alpha * np.log(beta) - a2 * np.log(b2))


def _rule_normal_update(prior_dist, data, known):
    from stochpylib.distributions import Normal
    x = _as_1d(data)
    sigma = known["sigma"]
    n = len(x)
    mu0, tau0 = prior_dist.mu, prior_dist.sigma
    prec0, prec_lik = 1.0 / tau0 ** 2, n / sigma ** 2
    post_prec = prec0 + prec_lik
    post_mean = (mu0 * prec0 + x.sum() / sigma ** 2) / post_prec
    return Normal(post_mean, np.sqrt(1.0 / post_prec))


def _rule_normal_predictive(post_dist, known):
    from stochpylib.distributions import Normal
    sigma = known["sigma"]
    return Normal(post_dist.mu, np.sqrt(post_dist.sigma ** 2 + sigma ** 2))


def _rule_normal_evidence(prior_dist, data, known):
    x = _as_1d(data)
    sigma = known["sigma"]
    n = len(x)
    mu0, tau0 = prior_dist.mu, prior_dist.sigma
    xbar = x.mean()
    var_pred = sigma ** 2 + n * tau0 ** 2
    return float(-n / 2 * np.log(2 * np.pi * sigma ** 2) - np.sum((x - xbar) ** 2) / (2 * sigma ** 2)
                 + 0.5 * np.log(sigma ** 2 / var_pred) - n * (xbar - mu0) ** 2 / (2 * var_pred))


def _rule_normal_variance_update(prior_dist, data, known):
    from stochpylib.distributions import InvGamma
    x = _as_1d(data)
    mu = known["mu"]
    n = len(x)
    return InvGamma(prior_dist.shape + n / 2.0, prior_dist.scale + 0.5 * np.sum((x - mu) ** 2))


def _rule_normal_variance_predictive(post_dist, known):
    from stochpylib.bayesian._common import _LocScaleT
    mu = known["mu"]
    a, b = post_dist.shape, post_dist.scale
    return _LocScaleT(2 * a, mu, np.sqrt(b / a))


def _rule_normal_variance_evidence(prior_dist, data, known):
    x = _as_1d(data)
    mu = known["mu"]
    n = len(x)
    a, b = prior_dist.shape, prior_dist.scale
    a2, b2 = a + n / 2.0, b + 0.5 * np.sum((x - mu) ** 2)
    return float(special.gammaln(a2) - special.gammaln(a) + a * np.log(b) - a2 * np.log(b2)
                 - n / 2.0 * np.log(2 * np.pi))


def _rule_niw_update(prior_dist, data, known):
    from stochpylib.bayesian._common import _NormalInverseGamma
    x = _as_1d(data)
    n = len(x)
    mu0, k0, a0, b0 = prior_dist.mu, prior_dist.kappa, prior_dist.alpha, prior_dist.beta
    xbar = x.mean() if n else 0.0
    k1 = k0 + n
    mu1 = (k0 * mu0 + n * xbar) / k1
    a1 = a0 + n / 2.0
    ss = np.sum((x - xbar) ** 2) if n else 0.0
    b1 = b0 + 0.5 * ss + k0 * n * (xbar - mu0) ** 2 / (2 * k1)
    return _NormalInverseGamma(mu1, k1, a1, b1)


def _rule_niw_predictive(post_dist, known):
    from stochpylib.bayesian._common import _LocScaleT
    mu1, k1, a1, b1 = post_dist.mu, post_dist.kappa, post_dist.alpha, post_dist.beta
    return _LocScaleT(2 * a1, mu1, np.sqrt(b1 * (k1 + 1) / (a1 * k1)))


def _rule_niw_evidence(prior_dist, data, known):
    x = _as_1d(data)
    n = len(x)
    mu0, k0, a0, b0 = prior_dist.mu, prior_dist.kappa, prior_dist.alpha, prior_dist.beta
    post = _rule_niw_update(prior_dist, data, known)
    a1, b1, k1 = post.alpha, post.beta, post.kappa
    return float(special.gammaln(a1) - special.gammaln(a0) + a0 * np.log(b0) - a1 * np.log(b1)
                 + 0.5 * np.log(k0 / k1) - n / 2.0 * np.log(2 * np.pi))


def _rule_categorical_update(prior_dist, data, known):
    from stochpylib.distributions import Dirichlet
    k = len(prior_dist.alpha)
    if known.get("counts"):
        counts = np.asarray(data, dtype=float)
    else:
        counts = _bincount_counts(data, k)
    return Dirichlet(prior_dist.alpha + counts)


def _rule_categorical_predictive(post_dist, known):
    from stochpylib.distributions import Multinomial
    p = post_dist.alpha / post_dist.alpha.sum()
    return Multinomial(1, p)


def _rule_categorical_evidence(prior_dist, data, known):
    from stochpylib.bayesian._common import _log_multibeta
    k = len(prior_dist.alpha)
    if known.get("counts"):
        counts = np.asarray(data, dtype=float)
        extra = float(special.gammaln(counts.sum() + 1) - np.sum(special.gammaln(counts + 1)))
    else:
        counts = _bincount_counts(data, k)
        extra = 0.0
    return extra + _log_multibeta(prior_dist.alpha + counts) - _log_multibeta(prior_dist.alpha)


def _rule_mvnormal_update(prior_dist, data, known):
    from stochpylib.distributions import MultivariateNormal
    X = _as_2d(data)
    n = X.shape[0]
    Sigma = np.asarray(known["cov"], dtype=float)
    S0inv = np.linalg.inv(prior_dist.cov_matrix)
    Sinv = np.linalg.inv(Sigma)
    Sn = np.linalg.inv(S0inv + n * Sinv)
    mn = Sn @ (S0inv @ prior_dist.mean_vec + Sinv @ X.sum(axis=0))
    return MultivariateNormal(mn, Sn)


def _rule_mvnormal_predictive(post_dist, known):
    from stochpylib.distributions import MultivariateNormal
    Sigma = np.asarray(known["cov"], dtype=float)
    return MultivariateNormal(post_dist.mean_vec, post_dist.cov_matrix + Sigma)


def _rule_mvnormal_evidence(prior_dist, data, known):
    X = _as_2d(data)
    n = X.shape[0]
    Sigma = np.asarray(known["cov"], dtype=float)
    xbar = X.mean(axis=0)
    lp_xbar = _mvn_logpdf(xbar, prior_dist.mean_vec, prior_dist.cov_matrix + Sigma / n)
    lp_data = sum(_mvn_logpdf(row, xbar, Sigma) for row in X)
    lp_xbar_given_xbar = _mvn_logpdf(xbar, xbar, Sigma / n)
    return float(lp_xbar + lp_data - lp_xbar_given_xbar)


_CONJUGATE_RULES = {
    "bernoulli": dict(update=_rule_bernoulli_update, predictive=_rule_bernoulli_predictive,
                       log_evidence=_rule_bernoulli_evidence),
    "binomial": dict(update=_rule_binomial_update, predictive=_rule_binomial_predictive,
                      log_evidence=_rule_binomial_evidence),
    "poisson": dict(update=_rule_poisson_update, predictive=_rule_poisson_predictive,
                     log_evidence=_rule_poisson_evidence),
    "exponential": dict(update=_rule_exponential_update, predictive=_rule_exponential_predictive,
                         log_evidence=_rule_exponential_evidence),
    "gamma": dict(update=_rule_gamma_update, predictive=_rule_gamma_predictive,
                  log_evidence=_rule_gamma_evidence),
    "normal": dict(update=_rule_normal_update, predictive=_rule_normal_predictive,
                    log_evidence=_rule_normal_evidence),
    "normal_variance": dict(update=_rule_normal_variance_update, predictive=_rule_normal_variance_predictive,
                             log_evidence=_rule_normal_variance_evidence),
    "normal_unknown_var": dict(update=_rule_niw_update, predictive=_rule_niw_predictive,
                                log_evidence=_rule_niw_evidence),
    "categorical": dict(update=_rule_categorical_update, predictive=_rule_categorical_predictive,
                         log_evidence=_rule_categorical_evidence),
    "mvnormal": dict(update=_rule_mvnormal_update, predictive=_rule_mvnormal_predictive,
                      log_evidence=_rule_mvnormal_evidence),
}


def conjugate_prior(family, **hyper):
    """Build the :class:`ConjugateFamily` for ``family`` (a name or a :class:`Likelihood`);
    with hyperparameters given, also sets ``.prior_``."""
    cf = ConjugateFamily(family)
    if hyper:
        cf.make_prior(**hyper)
    return cf


# ------------------------------------------------------------------------- posterior

def _is_conjugate_pair(pr, lik):
    if lik.family is None or lik.family not in _CONJUGATE_RULES:
        return False
    from stochpylib.distributions import Beta, Gamma, Normal, Dirichlet, MultivariateNormal, InvGamma
    from stochpylib.bayesian._common import _NormalInverseGamma
    dist = pr.dist
    if isinstance(dist, list) or dist == "flat" or dist is None:
        return False
    table = {
        "bernoulli": Beta, "binomial": Beta, "poisson": Gamma, "exponential": Gamma,
        "gamma": Gamma, "normal": Normal, "normal_variance": InvGamma,
        "normal_unknown_var": _NormalInverseGamma, "categorical": Dirichlet, "mvnormal": MultivariateNormal,
    }
    cls = table.get(lik.family)
    return cls is not None and isinstance(dist, cls)


def posterior(prior, likelihood, method="auto", *, grid=None, n_grid=201, n_samples=4000,
              n_warmup=None, sampler="slice", theta0=None, random_state=None, **kw):
    """Compute the posterior over the parameters of ``likelihood`` under ``prior``.

    ``method``: ``"auto"`` (conjugate if the pair matches, else grid for dim<=2, else
    mcmc), ``"conjugate"``, ``"grid"``, ``"laplace"``, ``"vi"``, ``"importance"``,
    ``"smc"``, or ``"mcmc"`` (``sampler`` in ``"slice"``/``"nuts"``/``"mh"``).
    """
    pr = globals()["prior"](prior) if not isinstance(prior, Prior) else prior
    lik = likelihood
    dim = pr.dim if pr.dim else lik.dim

    if method == "auto":
        if _is_conjugate_pair(pr, lik):
            method = "conjugate"
        elif dim is not None and dim <= 2:
            method = "grid"
        else:
            method = "mcmc"

    if method == "conjugate":
        if not _is_conjugate_pair(pr, lik):
            raise ValueError("prior/likelihood pair is not a recognized conjugate family")
        cf = ConjugateFamily(lik.family)
        post_dist = cf.update(pr.dist, lik.data, **lik.known)
        logZ = cf.log_evidence(pr.dist, lik.data, **lik.known)
        mean = np.atleast_1d(np.asarray(post_dist.mean(), dtype=float))
        try:
            var = np.atleast_1d(np.asarray(post_dist.var(), dtype=float))
            cov = np.diag(var) if var.ndim == 1 and dim == len(var) and dim > 1 else np.atleast_2d(var)
        except Exception:
            cov = None
        mode = mean  # unimodal exponential-family conjugate posteriors; exact mode not always closed-form
        return Posterior("conjugate", dim, dist=post_dist, mean_=mean, cov_=cov, mode_=mode,
                          log_evidence_=logZ, extras={"family": lik.family, "known": dict(lik.known)})

    if method == "grid":
        return _posterior_grid(pr, lik, grid, n_grid, dim)

    if method == "laplace":
        from stochpylib.bayesian.computation import LaplacePosterior
        t0 = theta0 if theta0 is not None else np.zeros(dim)
        return LaplacePosterior((pr, lik), t0).to_posterior()

    if method == "vi":
        from stochpylib.bayesian.computation import MFVariational
        return MFVariational((pr, lik), dim, random_state=random_state).to_posterior()

    if method == "importance":
        from stochpylib.bayesian.computation import ImportanceSamplingPosterior
        t0 = theta0 if theta0 is not None else np.zeros(dim)
        return ImportanceSamplingPosterior((pr, lik), theta0=t0, n=n_samples,
                                            random_state=random_state).to_posterior()

    if method == "smc":
        return _posterior_smc(pr, lik, n_samples, random_state)

    if method == "mcmc":
        return _posterior_mcmc(pr, lik, dim, sampler, n_samples, n_warmup, theta0, random_state)

    raise ValueError(f"unknown method: {method!r}")


def _grid_from_prior(pr, grid, n_grid, axis_dim):
    if grid is not None:
        return np.asarray(grid, dtype=float)
    if pr.dist == "flat" or pr.dist is None:
        raise ValueError("grid= is required for a flat or custom prior")
    d = pr.dist[axis_dim] if isinstance(pr.dist, list) else pr.dist
    lo, hi = float(d.ppf(1e-5)), float(d.ppf(1 - 1e-5))
    return np.linspace(lo, hi, n_grid)


def _posterior_grid(pr, lik, grid, n_grid, dim):
    if dim == 1:
        g = _grid_from_prior(pr, grid, n_grid, 0)
        log_post = np.array([pr.logpdf(np.array([t])) + lik.loglik(np.array([t])) for t in g])
        m = log_post.max()
        w = np.exp(log_post - m)
        widths = np.gradient(g)
        Z = np.sum(w * widths)
        dens = w / Z
        logZ = float(np.log(Z) + m)
        mean = np.array([np.sum(g * dens * widths)])
        var = np.sum((g - mean[0]) ** 2 * dens * widths)
        mode = np.array([g[np.argmax(dens)]])
        return Posterior("grid", 1, grid_=g, density_=dens, mean_=mean, cov_=[[var]],
                          mode_=mode, log_evidence_=logZ)
    if dim == 2:
        g0 = _grid_from_prior(pr, grid[0] if grid is not None else None, n_grid, 0)
        g1 = _grid_from_prior(pr, grid[1] if grid is not None else None, n_grid, 1)
        gx, gy = np.meshgrid(g0, g1)
        log_post = np.empty_like(gx)
        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                th = np.array([gx[i, j], gy[i, j]])
                log_post[i, j] = pr.logpdf(th) + lik.loglik(th)
        m = log_post.max()
        w = np.exp(log_post - m)
        dx = g0[1] - g0[0] if len(g0) > 1 else 1.0
        dy = g1[1] - g1[0] if len(g1) > 1 else 1.0
        Z = w.sum() * dx * dy
        dens = w / Z
        logZ = float(np.log(Z) + m)
        mean = np.array([np.sum(gx * dens) * dx * dy, np.sum(gy * dens) * dx * dy])
        c00 = np.sum((gx - mean[0]) ** 2 * dens) * dx * dy
        c11 = np.sum((gy - mean[1]) ** 2 * dens) * dx * dy
        c01 = np.sum((gx - mean[0]) * (gy - mean[1]) * dens) * dx * dy
        iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
        mode = np.array([gx[iy, ix], gy[iy, ix]])
        return Posterior("grid", 2, grid_=(gx, gy), density_=dens, mean_=mean,
                          cov_=[[c00, c01], [c01, c11]], mode_=mode, log_evidence_=logZ)
    raise ValueError("method='grid' only supports dim <= 2")


def _posterior_smc(pr, lik, n_particles, random_state):
    from stochpylib.advanced_mcmc import SequentialMonteCarlo
    smc = SequentialMonteCarlo(
        lambda th: pr.logpdf(th), lambda th: lik.loglik(th),
        lambda n, rng: pr.sample(n, random_state=rng), n_particles=n_particles)
    smc.sample(random_state=random_state)
    particles = smc.particles_ if hasattr(smc, "particles_") else smc.samples_
    weights = getattr(smc, "weights_", None)
    dim = particles.shape[1]
    mean = particles.mean(axis=0) if weights is None else np.average(particles, axis=0, weights=weights)
    cov = np.cov(particles, rowvar=False, aweights=weights) if dim > 1 else np.array(
        [[np.cov(particles[:, 0], aweights=weights)]])
    return Posterior("smc", dim, samples_=particles, weights_=weights, mean_=np.atleast_1d(mean),
                      cov_=np.atleast_2d(cov), log_evidence_=float(smc.log_evidence_),
                      extras={"engine": smc})


def _posterior_mcmc(pr, lik, dim, sampler, n_samples, n_warmup, theta0, random_state):
    from stochpylib.advanced_mcmc import SliceSampling, NoUTurnSampler, AdaptiveMetropolis

    def log_post(theta):
        return pr.logpdf(theta) + lik.loglik(theta)

    t0 = np.atleast_1d(np.asarray(theta0, dtype=float)) if theta0 is not None else np.ones(dim) * 0.1
    if sampler == "slice":
        mc = SliceSampling(log_post, n_samples=n_samples, n_warmup=n_warmup)
    elif sampler == "nuts":
        mc = NoUTurnSampler(log_post, n_samples=n_samples, n_warmup=n_warmup)
    elif sampler == "mh":
        mc = AdaptiveMetropolis(log_post, n_samples=n_samples, n_warmup=n_warmup)
    else:
        raise ValueError("sampler must be 'slice', 'nuts', or 'mh'")
    mc.sample(t0, random_state=random_state)
    samples = mc.get_samples()
    mean = samples.mean(axis=0)
    cov = np.atleast_2d(np.cov(samples, rowvar=False)) if dim > 1 else np.array([[samples.var()]])
    return Posterior("mcmc", dim, samples_=samples, mean_=mean, cov_=cov,
                      log_evidence_=None, extras={"sampler": mc, "acceptance_rate": mc.acceptance_rate_})


# ---------------------------------------------------------------------- bayes_update

def bayes_update(prior, data, family=None, **known):
    """Sequential conjugate update: returns the posterior distribution (same class as
    ``prior``). ``prior`` may be a distribution, an ``_NormalInverseGamma``, or a
    conjugate ``Posterior`` (uses ``.dist`` and its stored family)."""
    if isinstance(prior, Posterior):
        fam = family or prior.extras.get("family")
        dist = prior.dist
    else:
        dist = prior
        fam = family
    if fam is None:
        raise ValueError("family is required (or pass a conjugate Posterior)")
    cf = ConjugateFamily(fam)
    return cf.update(dist, data, **known)


# --------------------------------------------------------------- posterior_predictive

def posterior_predictive(post, likelihood=None, *, n_samples=4000, random_state=None, **known):
    """Posterior-predictive distribution: closed form for a conjugate posterior,
    otherwise a Monte Carlo mixture returned as :class:`EmpiricalPredictive`."""
    if isinstance(post, Posterior) and post.method == "conjugate":
        merged = {**post.extras.get("known", {}), **known}
        cf = ConjugateFamily(post.extras["family"])
        pred = cf.predictive(post.dist, **merged)
        if pred is not None:
            return pred
    dist = post.dist if isinstance(post, Posterior) else post
    fam = known.pop("family", None)
    if fam is None and isinstance(post, Posterior):
        fam = post.extras.get("family")
    if fam is not None:
        merged = {**(post.extras.get("known", {}) if isinstance(post, Posterior) else {}), **known}
        cf = ConjugateFamily(fam)
        pred = cf.predictive(dist, **merged)
        if pred is not None:
            return pred
    if likelihood is None:
        raise ValueError("likelihood is required for a non-conjugate predictive")
    if likelihood.family is None:
        raise ValueError("posterior_predictive requires a built-in likelihood family with a sampler")
    thetas = post.sample(n_samples, random_state=random_state) if isinstance(post, Posterior) else \
        np.atleast_2d(post)
    rng = _rng(random_state)
    draws = np.array([likelihood.sample(th, size=1, random_state=rng)[0] for th in thetas])
    from stochpylib.bayesian._result import EmpiricalPredictive
    return EmpiricalPredictive(draws)


# ------------------------------------------------------------------------- evidence

def evidence(prior, likelihood, method="auto", *, grid=None, n_samples=4000,
             random_state=None, **kw):
    """Log marginal likelihood ``log p(data)``. ``method``: ``"auto"`` (conjugate > grid
    (dim<=2) > laplace), ``"conjugate"``, ``"grid"``, ``"laplace"``, ``"importance"``,
    ``"smc"``."""
    pr = globals()["prior"](prior) if not isinstance(prior, Prior) else prior
    lik = likelihood
    dim = pr.dim if pr.dim else lik.dim

    if method == "auto":
        if _is_conjugate_pair(pr, lik):
            method = "conjugate"
        elif dim is not None and dim <= 2:
            method = "grid"
        else:
            method = "laplace"

    if method == "conjugate":
        if not _is_conjugate_pair(pr, lik):
            raise ValueError("prior/likelihood pair is not a recognized conjugate family")
        cf = ConjugateFamily(lik.family)
        return cf.log_evidence(pr.dist, lik.data, **lik.known)

    if method == "grid":
        if dim is None or dim > 2:
            raise ValueError("method='grid' only supports dim <= 2")
        return _posterior_grid(pr, lik, grid, kw.get("n_grid", 201), dim).log_evidence_

    if method == "laplace":
        from stochpylib.bayesian.computation import LaplacePosterior
        t0 = kw.get("theta0", np.zeros(dim))
        return LaplacePosterior((pr, lik), t0).log_evidence_

    if method == "importance":
        from stochpylib.bayesian.computation import ImportanceSamplingPosterior
        t0 = kw.get("theta0", np.zeros(dim))
        isp = ImportanceSamplingPosterior((pr, lik), theta0=t0, n=n_samples, random_state=random_state)
        return isp.log_evidence_

    if method == "smc":
        return _posterior_smc(pr, lik, n_samples, random_state).log_evidence_

    raise ValueError(f"unknown method: {method!r}")
