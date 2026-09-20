"""Bayesian models: linear/logistic regression, naive Bayes, hierarchical normal
models, finite mixtures, discrete Bayesian networks, Dirichlet-process mixtures."""

import numpy as np
from scipy import special

from stochpylib.bayesian._common import _as_1d, _as_2d, _cholesky_psd, _rng
from stochpylib.bayesian._common import _LocScaleT

__all__ = ["BayesianLinear", "BayesianLogistic", "NaiveBayes", "HierarchicalModel",
           "MixtureModel", "BayesianNetwork", "DirichletProcess"]


# ============================================================================ BayesianLinear

class BayesianLinear:
    """Conjugate Normal-Inverse-Gamma Bayesian linear regression:
    ``beta | sigma2 ~ N(m0, sigma2*V0)``, ``sigma2 ~ InvGamma(a0, b0)``."""

    def __init__(self, prior_precision=1e-6, prior_mean=None, prior_cov=None, a0=1e-2,
                 b0=1e-2, fit_intercept=True):
        self.prior_precision = float(prior_precision)
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.fit_intercept = bool(fit_intercept)

    def _design(self, X):
        X = _as_2d(X, "X")
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y):
        Xd = self._design(X)
        y = _as_1d(y, "y")
        n, p = Xd.shape
        V0 = (np.eye(p) / self.prior_precision if self.prior_cov is None
              else np.atleast_2d(np.asarray(self.prior_cov, dtype=float)))
        m0 = np.zeros(p) if self.prior_mean is None else np.asarray(self.prior_mean, dtype=float)
        V0inv = np.linalg.inv(V0)

        XtX = Xd.T @ Xd
        Vn_inv = V0inv + XtX
        L = _cholesky_psd(Vn_inv)
        Vn = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(p)))
        Vn = 0.5 * (Vn + Vn.T)
        mn = Vn @ (V0inv @ m0 + Xd.T @ y)
        an = self.a0 + n / 2.0
        bn = self.b0 + 0.5 * (y @ y + m0 @ V0inv @ m0 - mn @ Vn_inv @ mn)

        self.V0_, self.m0_ = V0, m0
        self.V_n_, self.a_n_, self.b_n_ = Vn, an, bn
        self.coef_ = mn
        self.mean_ = mn
        self.sigma2_ = bn / (an - 1.0) if an > 1 else np.inf
        self.cov_params_ = self.sigma2_ * Vn
        self.std_errors_ = np.sqrt(np.diag(self.cov_params_))
        self.n_obs_ = n
        self.p_ = p
        self._Xd_, self._y_ = Xd, y

        _, logdet_Vn = np.linalg.slogdet(Vn)
        _, logdet_V0 = np.linalg.slogdet(V0)
        self.log_evidence_ = float(
            special.gammaln(an) - special.gammaln(self.a0) + self.a0 * np.log(self.b0)
            - an * np.log(bn) + 0.5 * (logdet_Vn - logdet_V0) - n / 2.0 * np.log(2 * np.pi))
        self.loglik_ = self.log_likelihood_at_mean(Xd, y)
        return self

    def log_likelihood_at_mean(self, Xd, y):
        resid = y - Xd @ self.coef_
        sigma2 = self.sigma2_
        n = len(y)
        return float(-n / 2.0 * np.log(2 * np.pi * sigma2) - np.sum(resid ** 2) / (2 * sigma2))

    def predictive(self, X_new):
        Xd = self._design(X_new)
        mean = Xd @ self.coef_
        scale2 = (self.b_n_ / self.a_n_) * (1.0 + np.einsum("ij,jk,ik->i", Xd, self.V_n_, Xd))
        return [_LocScaleT(2 * self.a_n_, float(mean[i]), float(np.sqrt(scale2[i])))
                for i in range(Xd.shape[0])]

    def predict(self, X_new, return_std=False):
        Xd = self._design(X_new)
        mean = Xd @ self.coef_
        if not return_std:
            return mean
        scale2 = (self.b_n_ / self.a_n_) * (1.0 + np.einsum("ij,jk,ik->i", Xd, self.V_n_, Xd))
        df = 2 * self.a_n_
        var = scale2 * (df / (df - 2)) if df > 2 else np.full_like(scale2, np.inf)
        return mean, np.sqrt(var)

    def predict_interval(self, X_new, level=0.95):
        preds = self.predictive(X_new)
        alpha2 = (1 - level) / 2
        return [(float(p.ppf(alpha2)), float(p.ppf(1 - alpha2))) for p in preds]

    def credible_interval(self, level=0.95):
        alpha2 = (1 - level) / 2
        se = self.std_errors_
        df = 2 * self.a_n_
        from stochpylib.distributions import Student_t
        tcrit = float(Student_t(df).ppf(1 - alpha2))
        return [(float(self.coef_[j] - tcrit * se[j]), float(self.coef_[j] + tcrit * se[j]))
                for j in range(len(self.coef_))]

    def sample(self, n=1, random_state=None):
        rng = _rng(random_state)
        sigma2 = self.b_n_ / rng.gamma(self.a_n_, 1.0, size=n)
        L = _cholesky_psd(self.V_n_)
        z = rng.standard_normal((n, self.p_))
        beta = self.coef_[None, :] + np.sqrt(sigma2)[:, None] * (z @ L.T)
        return beta, sigma2

    def pointwise_log_lik(self, samples):
        """``(S, n)`` pointwise log-likelihood on the training data for WAIC/LOO, from
        ``sample()``'s ``(beta, sigma2)`` draws (or a bare ``(S, p)`` beta array at the
        fitted ``sigma2_``)."""
        if isinstance(samples, tuple):
            beta, sigma2 = samples
        else:
            beta = np.atleast_2d(samples)
            sigma2 = np.full(len(beta), self.sigma2_)
        resid = self._y_[None, :] - beta @ self._Xd_.T  # (S, n)
        return -0.5 * np.log(2 * np.pi * sigma2)[:, None] - 0.5 * resid ** 2 / sigma2[:, None]

    def summary(self):
        se = self.std_errors_
        return {"coef": self.coef_.tolist(), "std_errors": se.tolist(),
                "sigma2": float(self.sigma2_), "log_evidence": self.log_evidence_,
                "credible_interval_95": self.credible_interval(0.95)}

    def __repr__(self):
        return f"BayesianLinear(p={getattr(self, 'p_', '?')}, fit_intercept={self.fit_intercept})"


# =========================================================================== BayesianLogistic

def _sigmoid(z):
    return special.expit(z)


class BayesianLogistic:
    """Bayesian logistic regression with a Gaussian prior ``N(m0, prior_var*I)`` on the
    coefficients. ``method``: ``"laplace"`` (default, MAP + Gaussian curvature),
    ``"ep"`` (expectation propagation via :func:`~stochpylib.bayesian.computation.EP_Posterior`
    on the probit link -- the closest tractable EP site to a logistic one), ``"mcmc"``
    (NUTS with the analytic gradient), or ``"vi"``."""

    def __init__(self, prior_var=1.0, prior_mean=None, fit_intercept=True, method="laplace",
                 n_samples=2000, n_warmup=1000, random_state=None, max_iter=100, tol=1e-8):
        self.prior_var = float(prior_var)
        self.prior_mean = prior_mean
        self.fit_intercept = bool(fit_intercept)
        self.method = method
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_warmup)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _design(self, X):
        X = _as_2d(X, "X")
        return np.column_stack([np.ones(X.shape[0]), X]) if self.fit_intercept else X

    def _log_post_and_grad(self, Xd, y, m0, prior_prec):
        def log_post(beta):
            z = Xd @ beta
            ll = np.sum(y * z - np.logaddexp(0.0, z))
            lp = -0.5 * prior_prec * np.sum((beta - m0) ** 2)
            return ll + lp

        def grad(beta):
            p = _sigmoid(Xd @ beta)
            return Xd.T @ (y - p) - prior_prec * (beta - m0)

        return log_post, grad

    def _map_newton(self, Xd, y, m0, prior_prec):
        beta = m0.copy()
        p_dim = Xd.shape[1]
        for _ in range(self.max_iter):
            p = _sigmoid(Xd @ beta)
            grad = Xd.T @ (y - p) - prior_prec * (beta - m0)
            W = p * (1 - p)
            H = -(Xd.T * W) @ Xd - prior_prec * np.eye(p_dim)
            step = np.linalg.solve(-H, grad)
            beta_new = beta + step
            if np.max(np.abs(step)) < self.tol:
                beta = beta_new
                break
            beta = beta_new
        p = _sigmoid(Xd @ beta)
        W = p * (1 - p)
        H = -(Xd.T * W) @ Xd - prior_prec * np.eye(p_dim)
        return beta, H

    def fit(self, X, y):
        Xd = self._design(X)
        y = _as_1d(y, "y")
        self.classes_ = np.unique(y)
        if not set(self.classes_.tolist()) <= {0.0, 1.0}:
            raise ValueError("y must be binary in {0, 1}")
        n, p = Xd.shape
        m0 = np.zeros(p) if self.prior_mean is None else np.asarray(self.prior_mean, dtype=float)
        prior_prec = 1.0 / self.prior_var

        beta_hat, H = self._map_newton(Xd, y, m0, prior_prec)
        cov_laplace = np.linalg.inv(-H)
        cov_laplace = 0.5 * (cov_laplace + cov_laplace.T)
        log_post, grad = self._log_post_and_grad(Xd, y, m0, prior_prec)
        _, logdet_negH = np.linalg.slogdet(-H)
        self.map_ = beta_hat
        self.log_evidence_laplace_ = float(
            log_post(beta_hat) + p / 2.0 * np.log(2 * np.pi) - 0.5 * logdet_negH)

        if self.method == "laplace":
            self.coef_, self.cov_ = beta_hat, cov_laplace
            self.log_evidence_ = self.log_evidence_laplace_
            self.samples_ = None
        elif self.method == "ep":
            from stochpylib.bayesian.computation import EP_Posterior
            signed = np.where(y == 1, 1.0, -1.0)

            def log_site(f, i):
                # true logistic site (not a probit stand-in): EP's Gauss-Hermite moment
                # matching works against any analytic site, not only a Gaussian-conjugate one
                return -np.logaddexp(0.0, -signed[i] * f)

            ep = EP_Posterior(m0, np.eye(p) * self.prior_var, Xd, log_site)
            self.coef_, self.cov_ = ep.mean_, ep.cov_
            self.log_evidence_ = ep.log_evidence_
            self.samples_ = None
            self._ep_ = ep
        elif self.method == "mcmc":
            from stochpylib.advanced_mcmc import NoUTurnSampler
            nuts = NoUTurnSampler(log_post, grad, n_samples=self.n_samples,
                                   n_warmup=self.n_warmup)
            nuts.sample(beta_hat, random_state=self.random_state)
            self.samples_ = nuts.get_samples()
            self.coef_ = self.samples_.mean(axis=0)
            self.cov_ = np.atleast_2d(np.cov(self.samples_, rowvar=False))
            self.log_evidence_ = None
            self._mcmc_ = nuts
        elif self.method == "vi":
            from stochpylib.bayesian.computation import MFVariational
            vi = MFVariational(log_post, p, grad_log_prob=grad, random_state=self.random_state)
            self.coef_, self.cov_ = vi.mean_, vi.cov_
            self.log_evidence_ = vi.log_evidence_
            self.samples_ = None
            self._vi_ = vi
        else:
            raise ValueError("method must be 'laplace', 'ep', 'mcmc', or 'vi'")

        self.n_obs_, self.p_ = n, p
        self._Xd_, self._y_ = Xd, y
        self.n_iter_ = self.max_iter
        return self

    def _linear_predictor_stats(self, Xd):
        mean = Xd @ self.coef_
        var = np.einsum("ij,jk,ik->i", Xd, self.cov_, Xd)
        return mean, var

    def predict_proba(self, X_new, method="probit"):
        Xd = self._design(X_new)
        if method == "mc" or self.samples_ is not None:
            samples = self.samples_ if self.samples_ is not None else \
                np.random.default_rng(0).multivariate_normal(self.coef_, self.cov_, size=2000)
            probs = _sigmoid(samples @ Xd.T).mean(axis=0)
            return np.column_stack([1 - probs, probs])
        mean, var = self._linear_predictor_stats(Xd)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * var / 8.0)
        probs = _sigmoid(kappa * mean)
        return np.column_stack([1 - probs, probs])

    def predict(self, X_new):
        return (self.predict_proba(X_new)[:, 1] >= 0.5).astype(float)

    def credible_interval(self, level=0.95):
        alpha2 = (1 - level) / 2
        z = float(special.ndtri(1 - alpha2))
        se = np.sqrt(np.diag(self.cov_))
        return [(float(self.coef_[j] - z * se[j]), float(self.coef_[j] + z * se[j]))
                for j in range(len(self.coef_))]

    def sample(self, n=1, random_state=None):
        if self.samples_ is not None:
            rng = _rng(random_state)
            idx = rng.integers(0, len(self.samples_), size=n)
            return self.samples_[idx]
        rng = _rng(random_state)
        L = _cholesky_psd(self.cov_)
        return self.coef_ + rng.standard_normal((n, self.p_)) @ L.T

    def pointwise_log_lik(self, samples):
        z = np.atleast_2d(samples) @ self._Xd_.T
        y = self._y_[None, :]
        return y * z - np.logaddexp(0.0, z)

    def summary(self):
        return {"coef": self.coef_.tolist(), "std_errors": np.sqrt(np.diag(self.cov_)).tolist(),
                "log_evidence": self.log_evidence_, "method": self.method,
                "credible_interval_95": self.credible_interval(0.95)}

    def __repr__(self):
        return f"BayesianLogistic(method={self.method!r})"


# =============================================================================== NaiveBayes

class NaiveBayes:
    """Naive Bayes with a Bayesian (posterior-mean) point estimate of every parameter:
    Dirichlet(alpha) class prior, and per-feature conjugate posteriors --
    Beta(alpha,alpha) for ``"bernoulli"``, Dirichlet(alpha) for ``"multinomial"``, a
    variance-floored empirical mean/var for ``"gaussian"``."""

    def __init__(self, distribution="gaussian", alpha=1.0, class_prior=None, var_smoothing=1e-9):
        if distribution not in ("gaussian", "bernoulli", "multinomial"):
            raise ValueError("distribution must be 'gaussian', 'bernoulli', or 'multinomial'")
        self.distribution = distribution
        self.alpha = float(alpha)
        self.class_prior = class_prior
        self.var_smoothing = float(var_smoothing)

    def fit(self, X, y):
        X = _as_2d(X, "X")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n, d = X.shape
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)

        if self.class_prior is not None:
            self.class_log_prior_ = np.log(np.asarray(self.class_prior, dtype=float))
        else:
            self.class_log_prior_ = np.log((counts + self.alpha) / (n + self.alpha * n_classes))

        if self.distribution == "gaussian":
            self.theta_ = np.zeros((n_classes, d))
            self.var_ = np.zeros((n_classes, d))
            global_var = X.var(axis=0) * self.var_smoothing
            for ci, c in enumerate(self.classes_):
                Xc = X[y == c]
                self.theta_[ci] = Xc.mean(axis=0) if len(Xc) else X.mean(axis=0)
                v = Xc.var(axis=0, ddof=0) if len(Xc) > 1 else np.zeros(d)
                self.var_[ci] = v + global_var
        elif self.distribution == "bernoulli":
            self.theta_ = np.zeros((n_classes, d))
            for ci, c in enumerate(self.classes_):
                Xc = X[y == c]
                self.theta_[ci] = (Xc.sum(axis=0) + self.alpha) / (len(Xc) + 2 * self.alpha)
        else:  # multinomial
            self.theta_ = np.zeros((n_classes, d))
            for ci, c in enumerate(self.classes_):
                Xc = X[y == c]
                totals = Xc.sum()
                self.theta_[ci] = (Xc.sum(axis=0) + self.alpha) / (totals + self.alpha * d)

        self.feature_log_prob_ = np.log(np.clip(self.theta_, 1e-300, None)) \
            if self.distribution != "gaussian" else None
        self.class_count_ = counts
        self.n_features_ = d
        return self

    def _joint_log_likelihood(self, X):
        X = _as_2d(X, "X")
        n_classes = len(self.classes_)
        jll = np.tile(self.class_log_prior_, (X.shape[0], 1))
        if self.distribution == "gaussian":
            for ci in range(n_classes):
                var = self.var_[ci]
                diff = X - self.theta_[ci]
                jll[:, ci] += -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(diff ** 2 / var, axis=1)
        elif self.distribution == "bernoulli":
            for ci in range(n_classes):
                p = self.theta_[ci]
                jll[:, ci] += np.sum(X * np.log(p) + (1 - X) * np.log(1 - p), axis=1)
        else:
            for ci in range(n_classes):
                lp = self.feature_log_prob_[ci]
                jll[:, ci] += X @ lp
        return jll

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        from stochpylib.bayesian._common import _logsumexp
        norm = _logsumexp(jll, axis=1)
        return jll - norm[:, None]

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self):
        return f"NaiveBayes(distribution={self.distribution!r})"


# ========================================================================= HierarchicalModel

class HierarchicalModel:
    """Two-level normal-normal hierarchy: ``theta_j ~ N(mu, tau^2)``, ``mu ~ N(mu0,
    tau_mu^2)``. Mode A (``sigma=`` known per group, ``y`` = group effects, e.g. the
    eight-schools problem): Gibbs over ``(theta, mu, tau)``. Mode B (``groups=`` labels,
    raw observations, unknown shared ``sigma^2``): adds a conjugate ``sigma^2`` Gibbs step.
    ``tau_prior``: ``"half_cauchy"`` (default, slice-sampled), ``"inv_gamma"`` (conjugate
    Gibbs), or ``"fixed"`` (``tau=`` kwarg, no tau sampling).
    """

    def __init__(self, mu0=0.0, tau_mu=100.0, tau_prior="half_cauchy", tau_scale=5.0,
                 sigma_prior=(1e-2, 1e-2), n_samples=4000, n_warmup=1000, random_state=None):
        self.mu0 = float(mu0)
        self.tau_mu = float(tau_mu)
        self.tau_prior = tau_prior
        self.tau_scale = float(tau_scale)
        self.sigma_prior = sigma_prior
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_warmup)
        self.random_state = random_state

    def fit(self, y, sigma=None, groups=None, tau=None, tau_a0=1e-2, tau_b0=1e-2):
        from stochpylib.bayesian._common import _slice_step_1d
        rng = _rng(self.random_state)

        if groups is None:
            y = _as_1d(y, "y")
            if sigma is None:
                raise ValueError("mode A (no groups=) requires known sigma= per group")
            sigma = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape).copy()
            J = len(y)
            group_of = np.arange(J)
            counts = np.ones(J)
            ybar = y
            mode_b = False
        else:
            y = _as_1d(y, "y")
            groups = np.asarray(groups)
            uniq = np.unique(groups)
            J = len(uniq)
            group_of = np.searchsorted(uniq, groups)
            counts = np.array([(group_of == j).sum() for j in range(J)], dtype=float)
            ybar = np.array([y[group_of == j].mean() for j in range(J)])
            mode_b = True

        theta = ybar.copy()
        mu = float(np.mean(theta))
        tau_val = float(np.std(theta)) + 1e-3 if tau is None else float(tau)
        sigma2_shared = float(np.var(y)) + 1e-3 if mode_b else None

        n_total = self.n_warmup + self.n_samples
        theta_hist = np.empty((self.n_samples, J))
        mu_hist = np.empty(self.n_samples)
        tau_hist = np.empty(self.n_samples)
        sigma_hist = np.empty(self.n_samples) if mode_b else None

        for it in range(n_total):
            if mode_b:
                sig = np.sqrt(sigma2_shared)
                sigma_j = sig / np.sqrt(counts)
            else:
                sigma_j = sigma
            prec_j = 1.0 / sigma_j ** 2
            prec_theta = prec_j + 1.0 / tau_val ** 2
            mean_theta = (ybar * prec_j + mu / tau_val ** 2) / prec_theta
            theta = mean_theta + rng.standard_normal(J) / np.sqrt(prec_theta)

            prec_mu = J / tau_val ** 2 + 1.0 / self.tau_mu ** 2
            mean_mu = (np.sum(theta) / tau_val ** 2 + self.mu0 / self.tau_mu ** 2) / prec_mu
            mu = mean_mu + rng.standard_normal() / np.sqrt(prec_mu)

            if self.tau_prior == "fixed":
                pass
            elif self.tau_prior == "inv_gamma":
                a_n = tau_a0 + J / 2.0
                b_n = tau_b0 + 0.5 * np.sum((theta - mu) ** 2)
                tau2 = b_n / rng.gamma(a_n, 1.0)
                tau_val = float(np.sqrt(tau2))
            else:  # half_cauchy on tau, slice sample log(tau)
                def log_tau_post(log_t):
                    t = np.exp(log_t)
                    ll = -J * log_t - 0.5 * np.sum((theta - mu) ** 2) / t ** 2
                    lp = -np.log(self.tau_scale ** 2 + t ** 2) + log_t  # half-Cauchy + Jacobian
                    return ll + lp

                log_tau = _slice_step_1d(log_tau_post, np.log(max(tau_val, 1e-6)), w=1.0, rng=rng)
                tau_val = float(np.exp(log_tau))

            if mode_b:
                resid2 = np.sum((y - theta[group_of]) ** 2)
                a0s, b0s = self.sigma_prior
                a_n = a0s + len(y) / 2.0
                b_n = b0s + 0.5 * resid2
                sigma2_shared = float(b_n / rng.gamma(a_n, 1.0))

            if it >= self.n_warmup:
                k = it - self.n_warmup
                theta_hist[k] = theta
                mu_hist[k] = mu
                tau_hist[k] = tau_val
                if mode_b:
                    sigma_hist[k] = sigma2_shared

        self.theta_samples_ = theta_hist
        self.mu_samples_ = mu_hist
        self.tau_samples_ = tau_hist
        self.sigma_samples_ = sigma_hist
        self.theta_mean_ = theta_hist.mean(axis=0)
        self.theta_std_ = theta_hist.std(axis=0, ddof=1)
        self.mu_mean_ = float(mu_hist.mean())
        self.tau_mean_ = float(tau_hist.mean())
        self.groups_ = np.unique(groups) if groups is not None else None
        self._group_of_, self._y_, self._mode_b_ = group_of, y, mode_b
        if mode_b:
            eff_sigma2 = np.array([self.sigma_samples_.mean() / c for c in counts])
        else:
            eff_sigma2 = sigma ** 2
        self.shrinkage_ = eff_sigma2 / (eff_sigma2 + self.tau_mean_ ** 2)
        return self

    def summary(self):
        from stochpylib.advanced_mcmc import ESS
        out = {}
        for j in range(self.theta_samples_.shape[1]):
            col = self.theta_samples_[:, j]
            ess = float(ESS(col[None, :, None], method="mean"))
            out[f"theta_{j}"] = {"mean": float(col.mean()), "std": float(col.std(ddof=1)),
                                  "ci_95": (float(np.quantile(col, 0.025)), float(np.quantile(col, 0.975))),
                                  "ess": ess}
        out["mu"] = {"mean": self.mu_mean_, "std": float(self.mu_samples_.std(ddof=1))}
        out["tau"] = {"mean": self.tau_mean_, "std": float(self.tau_samples_.std(ddof=1))}
        return out

    def credible_interval(self, level=0.95):
        alpha2 = (1 - level) / 2
        out = [(float(np.quantile(self.theta_samples_[:, j], alpha2)),
                float(np.quantile(self.theta_samples_[:, j], 1 - alpha2)))
               for j in range(self.theta_samples_.shape[1])]
        return out

    def predict_new_group(self, n, random_state=None):
        rng = _rng(random_state)
        idx = rng.integers(0, len(self.mu_samples_), size=n)
        return self.mu_samples_[idx] + self.tau_samples_[idx] * rng.standard_normal(n)

    def pooled_estimate(self):
        return float(self.mu_mean_)

    def pointwise_log_lik(self):
        if self._mode_b_:
            sig2 = self.sigma_samples_[:, None]
            resid = self._y_[None, :] - self.theta_samples_[:, self._group_of_]
            return -0.5 * np.log(2 * np.pi * sig2) - 0.5 * resid ** 2 / sig2
        raise NotImplementedError("pointwise_log_lik requires mode B (groups= and unknown sigma)")

    def __repr__(self):
        return f"HierarchicalModel(J={self.theta_samples_.shape[1] if hasattr(self, 'theta_samples_') else '?'})"


# ============================================================================= MixtureModel

class MixtureModel:
    """Bayesian finite mixture model fit by collapsed-conditional Gibbs sampling.
    ``family="gaussian"``: Normal-Inverse-(Chi-squared/Wishart) base measure (1-D uses the
    NIG conjugate path, multi-D the NIW path); ``family="poisson"``: Gamma-Poisson
    (1-D counts only). ``relabel="order"`` sorts components by their mean's first
    coordinate each draw (a documented mitigation of label switching)."""

    def __init__(self, n_components, family="gaussian", alpha=1.0, n_samples=2000,
                 n_warmup=500, random_state=None, relabel="order"):
        self.n_components = int(n_components)
        self.family = family
        self.alpha = float(alpha)
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_warmup)
        self.random_state = random_state
        self.relabel = relabel

    def fit(self, X):
        from stochpylib.statistics import cluster_analysis
        rng = _rng(self.random_state)
        K = self.n_components

        if self.family == "poisson":
            x = _as_1d(X, "X")
            n = len(x)
            labels = _rng(0).integers(0, K, size=n) if K > 1 else np.zeros(n, dtype=int)
            a0, b0 = 1.0, 1.0
            weight_hist = np.empty((self.n_samples, K))
            rate_hist = np.empty((self.n_samples, K))
            ll_hist = np.empty((self.n_samples, n))
            for it in range(self.n_warmup + self.n_samples):
                counts = np.array([(labels == k).sum() for k in range(K)])
                pi = rng.dirichlet(self.alpha / K + counts)
                rates = np.empty(K)
                for k in range(K):
                    xs = x[labels == k]
                    an = a0 + xs.sum()
                    bn = b0 + len(xs)
                    rates[k] = rng.gamma(an, 1.0 / bn)
                log_p = np.log(np.clip(pi, 1e-300, None))[None, :] + \
                    (x[:, None] * np.log(np.clip(rates, 1e-300, None))[None, :] - rates[None, :])
                log_p -= log_p.max(axis=1, keepdims=True)
                p = np.exp(log_p)
                p /= p.sum(axis=1, keepdims=True)
                labels = np.array([rng.choice(K, p=p[i]) for i in range(n)])
                if it >= self.n_warmup:
                    k_idx = it - self.n_warmup
                    order = np.argsort(rates) if self.relabel == "order" else np.arange(K)
                    weight_hist[k_idx] = pi[order]
                    rate_hist[k_idx] = rates[order]
                    ll_hist[k_idx] = np.log(np.clip(
                        np.sum(pi[order][None, :] * np.exp(-rates[order][None, :])
                               * rates[order][None, :] ** x[:, None]
                               / special.gamma(x[:, None] + 1), axis=1), 1e-300, None))
            self.weights_ = weight_hist.mean(axis=0)
            self.means_ = rate_hist.mean(axis=0)
            self.covariances_ = None
            self.weight_samples_ = weight_hist
            self.mean_samples_ = rate_hist
            self.pointwise_log_lik_ = ll_hist
            self.log_lik_samples_ = ll_hist.sum(axis=1)
            self.labels_ = labels
            self._X_, self._rates_last_ = x, rate_hist[-1]
            return self

        X = _as_2d(X, "X")
        n, d = X.shape
        labels = cluster_analysis(X, K, method="kmeans", random_state=0).labels_ if K > 1 \
            else np.zeros(n, dtype=int)
        m0 = X.mean(axis=0)
        kappa0 = 0.01
        if d == 1:
            a0, b0 = 2.0, float(X.var()) + 1e-6
        else:
            nu0 = d + 2
            Psi0 = np.atleast_2d(np.cov(X, rowvar=False)) * (nu0 - d - 1) + 1e-6 * np.eye(d)

        weight_hist = np.empty((self.n_samples, K))
        mean_hist = np.empty((self.n_samples, K, d))
        cov_hist = np.empty((self.n_samples, K, d, d))
        ll_hist = np.empty((self.n_samples, n))
        means, covs = np.zeros((K, d)), np.array([np.eye(d)] * K)

        for it in range(self.n_warmup + self.n_samples):
            counts = np.array([(labels == k).sum() for k in range(K)])
            pi = rng.dirichlet(self.alpha / K + counts)
            for k in range(K):
                Xk = X[labels == k]
                nk = len(Xk)
                if d == 1:
                    xbar = Xk.mean() if nk else m0[0]
                    kn = kappa0 + nk
                    mn = (kappa0 * m0[0] + nk * xbar) / kn
                    an = a0 + nk / 2.0
                    ss = np.sum((Xk[:, 0] - xbar) ** 2) if nk else 0.0
                    bn = b0 + 0.5 * ss + kappa0 * nk * (xbar - m0[0]) ** 2 / (2 * kn)
                    sigma2 = bn / rng.gamma(an, 1.0)
                    mu = mn + rng.standard_normal() * np.sqrt(sigma2 / kn)
                    means[k] = [mu]
                    covs[k] = np.array([[sigma2]])
                else:
                    xbar = Xk.mean(axis=0) if nk else m0
                    kn = kappa0 + nk
                    mn = (kappa0 * m0 + nk * xbar) / kn
                    nun = nu0 + nk
                    S = (Xk - xbar).T @ (Xk - xbar) if nk else np.zeros((d, d))
                    Psin = Psi0 + S + (kappa0 * nk / kn) * np.outer(xbar - m0, xbar - m0)
                    from stochpylib.distributions import InverseWishart
                    Sigma_k = InverseWishart(nun, Psin).rvs(1, random_state=rng)
                    L = _cholesky_psd(Sigma_k / kn)
                    mu = mn + L @ rng.standard_normal(d)
                    means[k], covs[k] = mu, Sigma_k

            log_p = np.empty((n, K))
            for k in range(K):
                L = _cholesky_psd(covs[k])
                diff = X - means[k]
                y = np.linalg.solve(L, diff.T).T
                logdet = 2 * np.sum(np.log(np.diag(L)))
                log_p[:, k] = np.log(max(pi[k], 1e-300)) - 0.5 * (d * np.log(2 * np.pi) + logdet
                                                                    + np.sum(y ** 2, axis=1))
            log_p -= log_p.max(axis=1, keepdims=True)
            p = np.exp(log_p)
            p /= p.sum(axis=1, keepdims=True)
            labels = np.array([rng.choice(K, p=p[i]) for i in range(n)])

            if it >= self.n_warmup:
                k_idx = it - self.n_warmup
                order = np.argsort(means[:, 0]) if self.relabel == "order" else np.arange(K)
                weight_hist[k_idx] = pi[order]
                mean_hist[k_idx] = means[order]
                cov_hist[k_idx] = covs[order]
                mix_ll = np.empty(n)
                lp2 = np.empty((n, K))
                for kk, k in enumerate(order):
                    L = _cholesky_psd(covs[k])
                    diff = X - means[k]
                    yv = np.linalg.solve(L, diff.T).T
                    logdet = 2 * np.sum(np.log(np.diag(L)))
                    lp2[:, kk] = np.log(max(pi[k], 1e-300)) - 0.5 * (d * np.log(2 * np.pi) + logdet
                                                                       + np.sum(yv ** 2, axis=1))
                m = lp2.max(axis=1)
                mix_ll = m + np.log(np.sum(np.exp(lp2 - m[:, None]), axis=1))
                ll_hist[k_idx] = mix_ll

        self.weights_ = weight_hist.mean(axis=0)
        self.means_ = mean_hist.mean(axis=0)
        self.covariances_ = cov_hist.mean(axis=0)
        self.weight_samples_ = weight_hist
        self.mean_samples_ = mean_hist
        self.log_lik_samples_ = ll_hist.sum(axis=1)
        self.pointwise_log_lik_ = ll_hist
        self.labels_ = labels
        self._X_ = X
        return self

    def predict_proba(self, X):
        X = _as_2d(X, "X") if self.covariances_ is not None else _as_1d(X, "X")[:, None]
        n = X.shape[0]
        K = self.n_components
        if self.covariances_ is None:  # poisson
            x = X[:, 0]
            log_p = np.log(np.clip(self.weights_, 1e-300, None))[None, :] + \
                (x[:, None] * np.log(np.clip(self.means_, 1e-300, None))[None, :] - self.means_[None, :])
        else:
            log_p = np.empty((n, K))
            for k in range(K):
                d = X.shape[1]
                L = _cholesky_psd(self.covariances_[k])
                diff = X - self.means_[k]
                y = np.linalg.solve(L, diff.T).T
                logdet = 2 * np.sum(np.log(np.diag(L)))
                log_p[:, k] = np.log(max(self.weights_[k], 1e-300)) - 0.5 * (
                    d * np.log(2 * np.pi) + logdet + np.sum(y ** 2, axis=1))
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X):
        from stochpylib.bayesian._common import _logsumexp
        probs = self.predict_proba(X)
        # recompute joint log density (not just responsibilities) for score_samples
        X2 = _as_2d(X, "X") if self.covariances_ is not None else _as_1d(X, "X")[:, None]
        n, K = X2.shape[0], self.n_components
        if self.covariances_ is None:
            x = X2[:, 0]
            log_p = np.log(np.clip(self.weights_, 1e-300, None))[None, :] + \
                (x[:, None] * np.log(np.clip(self.means_, 1e-300, None))[None, :] - self.means_[None, :]
                 - special.gammaln(x[:, None] + 1))
        else:
            log_p = np.empty((n, K))
            for k in range(K):
                d = X2.shape[1]
                L = _cholesky_psd(self.covariances_[k])
                diff = X2 - self.means_[k]
                y = np.linalg.solve(L, diff.T).T
                logdet = 2 * np.sum(np.log(np.diag(L)))
                log_p[:, k] = np.log(max(self.weights_[k], 1e-300)) - 0.5 * (
                    d * np.log(2 * np.pi) + logdet + np.sum(y ** 2, axis=1))
        return _logsumexp(log_p, axis=1)

    def sample(self, n=1, random_state=None):
        rng = _rng(random_state)
        z = rng.choice(self.n_components, size=n, p=self.weights_)
        if self.covariances_ is None:
            return rng.poisson(self.means_[z])
        d = self.means_.shape[1]
        out = np.empty((n, d))
        for k in np.unique(z):
            idx = z == k
            L = _cholesky_psd(self.covariances_[k])
            out[idx] = self.means_[k] + rng.standard_normal((idx.sum(), d)) @ L.T
        return out

    def waic(self):
        from stochpylib.bayesian.selection import WAIC
        return WAIC(self.pointwise_log_lik_)

    def loo(self):
        from stochpylib.bayesian.selection import LOO_CV
        return LOO_CV(self.pointwise_log_lik_)

    def __repr__(self):
        return f"MixtureModel(n_components={self.n_components}, family={self.family!r})"


# =========================================================================== BayesianNetwork

class BayesianNetwork:
    """A discrete Bayesian network with exact inference (variable elimination) and a
    Dirichlet(alpha)-smoothed MLE fit from complete data."""

    def __init__(self):
        self.nodes = []
        self._states = {}
        self._parents = {}
        self._children = {}
        self.cpts_ = {}

    def add_node(self, name, states):
        if name in self._states:
            raise ValueError(f"node {name!r} already exists")
        self.nodes.append(name)
        self._states[name] = list(states)
        self._parents[name] = []
        self._children[name] = []

    def add_edge(self, parent, child):
        if parent not in self._states or child not in self._states:
            raise ValueError("both nodes must be added first")
        self._parents[child].append(parent)
        self._children[parent].append(child)
        try:
            self.topological_order()
        except ValueError:
            self._parents[child].remove(parent)
            self._children[parent].remove(child)
            raise ValueError(f"adding edge {parent}->{child} creates a cycle")

    def parents(self, node):
        return list(self._parents[node])

    def children(self, node):
        return list(self._children[node])

    def topological_order(self):
        visited, order, temp = set(), [], set()

        def visit(n):
            if n in visited:
                return
            if n in temp:
                raise ValueError("graph has a cycle")
            temp.add(n)
            for p in self._parents[n]:
                visit(p)
            temp.discard(n)
            visited.add(n)
            order.append(n)

        for n in self.nodes:
            visit(n)
        return order

    def markov_blanket(self, node):
        mb = set(self._parents[node])
        for c in self._children[node]:
            mb.add(c)
            mb.update(self._parents[c])
        mb.discard(node)
        return sorted(mb)

    def set_cpt(self, node, table):
        table = np.asarray(table, dtype=float)
        expected_shape = tuple(len(self._states[p]) for p in self._parents[node]) + \
            (len(self._states[node]),)
        if table.shape != expected_shape:
            raise ValueError(f"CPT for {node!r} must have shape {expected_shape}, got {table.shape}")
        if not np.allclose(table.sum(axis=-1), 1.0, atol=1e-6):
            raise ValueError(f"CPT rows for {node!r} must sum to 1")
        self.cpts_[node] = table

    def _state_index(self, node, value):
        return self._states[node].index(value)

    def joint_probability(self, assignment):
        p = 1.0
        for node in self.nodes:
            table = self.cpts_[node]
            pa_idx = tuple(self._state_index(pa, assignment[pa]) for pa in self._parents[node])
            p *= table[pa_idx + (self._state_index(node, assignment[node]),)]
        return float(p)

    def _factor_for(self, node):
        return (tuple(self._parents[node]) + (node,), self.cpts_[node].copy())

    def query(self, variables, evidence=None):
        evidence = evidence or {}
        factors = [self._factor_for(n) for n in self.nodes]
        # restrict factors to observed evidence values
        restricted = []
        for vars_, table in factors:
            for ev_var, ev_val in evidence.items():
                if ev_var in vars_:
                    axis = vars_.index(ev_var)
                    idx = self._state_index(ev_var, ev_val)
                    table = np.take(table, idx, axis=axis)
                    vars_ = tuple(v for v in vars_ if v != ev_var)
            restricted.append((vars_, table))

        elim_order = [n for n in self.nodes if n not in variables and n not in evidence]
        for var in elim_order:
            involved = [(v, t) for v, t in restricted if var in v]
            rest = [(v, t) for v, t in restricted if var not in v]
            if not involved:
                continue
            merged_vars = []
            for v, _ in involved:
                for name in v:
                    if name not in merged_vars:
                        merged_vars.append(name)
            merged = None
            for v, t in involved:
                shape = [len(self._states[name]) if name in v else 1 for name in merged_vars]
                t_full = np.reshape(np.transpose(t, [v.index(name) for name in merged_vars if name in v]), shape)
                merged = t_full if merged is None else merged * t_full
            sum_axis = merged_vars.index(var)
            new_table = np.sum(merged, axis=sum_axis)
            new_vars = tuple(name for name in merged_vars if name != var)
            restricted = rest + [(new_vars, new_table)]

        merged_vars = []
        for v, _ in restricted:
            for name in v:
                if name not in merged_vars:
                    merged_vars.append(name)
        merged = None
        for v, t in restricted:
            shape = [len(self._states[name]) if name in v else 1 for name in merged_vars]
            t_full = np.reshape(np.transpose(t, [v.index(name) for name in merged_vars if name in v]), shape)
            merged = t_full if merged is None else merged * t_full
        order = [merged_vars.index(v) for v in variables]
        result = np.transpose(merged, order + [i for i in range(len(merged_vars)) if i not in order])
        result = result.reshape([len(self._states[v]) for v in variables])
        result = result / result.sum()
        result = _CPTResult(result, [self._states[v] for v in variables])
        return result

    def map_query(self, variables, evidence=None):
        table = self.query(variables, evidence)
        idx = np.unravel_index(np.argmax(table), table.shape)
        return {v: self._states[v][i] for v, i in zip(variables, idx)}

    def sample(self, n, random_state=None):
        rng = _rng(random_state)
        order = self.topological_order()
        col = {node: i for i, node in enumerate(self.nodes)}
        assign_idx = np.empty((n, len(self.nodes)), dtype=int)
        for node in order:
            table = self.cpts_[node]
            pa = self._parents[node]
            if not pa:
                p = table
                assign_idx[:, col[node]] = rng.choice(len(self._states[node]), size=n, p=p)
            else:
                pa_cols = [col[p] for p in pa]
                for i in range(n):
                    idx = tuple(assign_idx[i, c] for c in pa_cols)
                    p = table[idx]
                    assign_idx[i, col[node]] = rng.choice(len(self._states[node]), p=p)
        return assign_idx

    def fit(self, data, alpha=1.0, nodes=None):
        data = np.asarray(data, dtype=int)
        nodes = nodes or self.nodes
        col = {node: i for i, node in enumerate(nodes)}
        for node in self.nodes:
            pa = self._parents[node]
            r = len(self._states[node])
            pa_sizes = [len(self._states[p]) for p in pa]
            q = int(np.prod(pa_sizes)) if pa_sizes else 1
            counts = np.zeros(tuple(pa_sizes) + (r,))
            node_col = col[node]
            pa_cols = [col[p] for p in pa]
            for row in data:
                idx = tuple(row[c] for c in pa_cols) + (row[node_col],)
                counts[idx] += 1
            table = (counts + alpha / (q * r)) / (counts.sum(axis=-1, keepdims=True) + alpha / q)
            self.cpts_[node] = table
        return self

    def log_likelihood(self, data):
        data = np.asarray(data, dtype=int)
        col = {node: i for i, node in enumerate(self.nodes)}
        total = 0.0
        for row in data:
            p = 1.0
            for node in self.nodes:
                pa = self._parents[node]
                pa_idx = tuple(row[col[p_]] for p_ in pa)
                p *= self.cpts_[node][pa_idx + (row[col[node]],)]
            total += np.log(max(p, 1e-300))
        return float(total)

    def log_marginal_likelihood(self, data, alpha=1.0):
        """BDeu score: sum over nodes of the Dirichlet-multinomial marginal likelihood
        for that node's local CPT, integrating out the CPT parameters."""
        from stochpylib.bayesian._common import _log_multibeta
        data = np.asarray(data, dtype=int)
        col = {node: i for i, node in enumerate(self.nodes)}
        total = 0.0
        for node in self.nodes:
            pa = self._parents[node]
            r = len(self._states[node])
            pa_sizes = [len(self._states[p]) for p in pa]
            q = int(np.prod(pa_sizes)) if pa_sizes else 1
            counts = np.zeros(tuple(pa_sizes) + (r,))
            node_col = col[node]
            pa_cols = [col[p] for p in pa]
            for row in data:
                idx = tuple(row[c] for c in pa_cols) + (row[node_col],)
                counts[idx] += 1
            counts_flat = counts.reshape(q, r)
            for j in range(q):
                a_j = np.full(r, alpha / (q * r))
                total += _log_multibeta(a_j + counts_flat[j]) - _log_multibeta(a_j)
        return float(total)

    def __repr__(self):
        return f"BayesianNetwork(nodes={self.nodes})"


class _CPTResult(np.ndarray):
    """A probability table over named discrete variables (thin ndarray subclass so
    ``query()`` results can be indexed like a plain array but still remember the state
    labels via ``.states``)."""

    def __new__(cls, array, states):
        obj = np.asarray(array, dtype=float).view(cls)
        obj.states = states
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.states = getattr(obj, "states", None)


# ========================================================================== DirichletProcess

class DirichletProcess:
    """A Dirichlet process ``DP(alpha, base)`` -- stick-breaking, the Chinese restaurant
    process, and (via ``fit``) a collapsed Gibbs (Neal 2000, Algorithm 3) Gaussian DP
    mixture with a conjugate Normal-Inverse-Gamma/-Wishart base measure."""

    def __init__(self, alpha=1.0, base=None, truncation=100):
        self.alpha = float(alpha)
        from stochpylib.distributions import Normal
        self.base = base if base is not None else Normal(0.0, 1.0)
        self.truncation = int(truncation)

    def stick_breaking(self, random_state=None, n_atoms=None):
        rng = _rng(random_state)
        K = n_atoms or self.truncation
        beta = rng.beta(1.0, self.alpha, size=K)
        remaining = np.cumprod(np.concatenate([[1.0], 1 - beta[:-1]]))
        weights = beta * remaining
        weights[-1] += max(0.0, 1.0 - weights.sum())  # truncated stick gets the remainder
        atoms = np.atleast_1d(self.base.rvs(K, random_state=rng))
        return weights, atoms

    def crp(self, n, random_state=None):
        rng = _rng(random_state)
        labels = np.zeros(n, dtype=int)
        counts = [1]
        labels[0] = 0
        for i in range(1, n):
            probs = np.array(counts + [self.alpha], dtype=float)
            probs /= probs.sum()
            k = rng.choice(len(probs), p=probs)
            if k == len(counts):
                counts.append(1)
            else:
                counts[k] += 1
            labels[i] = k
        return labels

    def expected_clusters(self, n):
        return float(np.sum(self.alpha / (self.alpha + np.arange(n))))

    def sample_prior(self, n, random_state=None):
        rng = _rng(random_state)
        labels = self.crp(n, random_state=rng)
        K = labels.max() + 1
        atoms = np.atleast_1d(self.base.rvs(K, random_state=rng))
        return atoms[labels], labels

    def fit(self, X, n_samples=1000, n_warmup=500, learn_alpha=False, alpha_prior=(1.0, 1.0),
            random_state=None):
        rng = _rng(random_state)
        X = np.asarray(X, dtype=float)
        d = 1 if X.ndim == 1 else X.shape[1]
        X2 = X[:, None] if X.ndim == 1 else X
        n = X2.shape[0]
        m0 = X2.mean(axis=0)
        kappa0 = 0.01
        if d == 1:
            a0, b0 = 2.0, float(X2.var()) + 1e-6
        else:
            nu0 = d + 2
            Psi0 = np.atleast_2d(np.cov(X2, rowvar=False)) * (nu0 - d - 1) + 1e-6 * np.eye(d)

        labels = np.zeros(n, dtype=int)
        alpha = self.alpha
        label_hist = np.empty((n_samples, n), dtype=int)
        k_hist = np.empty(n_samples, dtype=int)
        alpha_hist = np.empty(n_samples) if learn_alpha else None

        def marginal_logpdf(x, members):
            """log p(x | members of a cluster), integrating out (mu, sigma2/Sigma)."""
            nk = len(members)
            if d == 1:
                x = float(x[0]) if np.ndim(x) else float(x)
                xbar = members.mean() if nk else m0[0]
                kn = kappa0 + nk
                mn = (kappa0 * m0[0] + nk * xbar) / kn
                an = a0 + nk / 2.0
                ss = np.sum((members - xbar) ** 2) if nk else 0.0
                bn = b0 + 0.5 * ss + kappa0 * nk * (xbar - m0[0]) ** 2 / (2 * kn)
                t_dist = _LocScaleT(2 * an, mn, float(np.sqrt(bn * (kn + 1) / (an * kn))))
                return float(np.log(max(float(t_dist.pdf(x)), 1e-300)))
            xbar = members.mean(axis=0) if nk else m0
            kn = kappa0 + nk
            mn = (kappa0 * m0 + nk * xbar) / kn
            nun = nu0 + nk
            S = (members - xbar).T @ (members - xbar) if nk else np.zeros((d, d))
            Psin = Psi0 + S + (kappa0 * nk / kn) * np.outer(xbar - m0, xbar - m0)
            from stochpylib.distributions import MultivariateT
            df = nun - d + 1
            shape = Psin * (kn + 1) / (kn * df)
            mt = MultivariateT(df, mn, shape)
            return float(np.log(max(float(mt.pdf(x)), 1e-300)))

        for it in range(n_warmup + n_samples):
            for i in range(n):
                labels[i] = -1  # remove i
                uniq = np.unique(labels[labels >= 0])
                probs, cand = [], []
                for k in uniq:
                    members = X2[labels == k]
                    nk = len(members)
                    lp = np.log(nk) + marginal_logpdf(X2[i], members)
                    probs.append(lp)
                    cand.append(k)
                lp_new = np.log(alpha) + marginal_logpdf(X2[i], X2[0:0])
                probs.append(lp_new)
                cand.append(-2)  # sentinel for "new cluster"
                probs = np.array(probs)
                probs -= probs.max()
                w = np.exp(probs)
                w /= w.sum()
                choice = rng.choice(len(w), p=w)
                if cand[choice] == -2:
                    new_k = 0
                    existing = np.unique(labels[labels >= 0])
                    while new_k in existing:
                        new_k += 1
                    labels[i] = new_k
                else:
                    labels[i] = cand[choice]

            if learn_alpha:
                K = len(np.unique(labels))
                eta = rng.beta(alpha + 1, n)
                a_pr, b_pr = alpha_prior
                pi_mix = (a_pr + K - 1) / (a_pr + K - 1 + n * (b_pr - np.log(eta)))
                if rng.uniform() < pi_mix:
                    alpha = rng.gamma(a_pr + K, 1.0 / (b_pr - np.log(eta)))
                else:
                    alpha = rng.gamma(a_pr + K - 1, 1.0 / (b_pr - np.log(eta)))

            if it >= n_warmup:
                k_idx = it - n_warmup
                # relabel to consecutive ints 0..K-1 in order of first appearance
                remap, next_id = {}, 0
                relabeled = np.empty(n, dtype=int)
                for i in range(n):
                    if labels[i] not in remap:
                        remap[labels[i]] = next_id
                        next_id += 1
                    relabeled[i] = remap[labels[i]]
                label_hist[k_idx] = relabeled
                k_hist[k_idx] = next_id
                if learn_alpha:
                    alpha_hist[k_idx] = alpha

        self.label_samples_ = label_hist
        self.n_clusters_samples_ = k_hist
        self.n_clusters_ = int(np.bincount(k_hist).argmax())
        self.labels_ = label_hist[-1]
        self.alpha_samples_ = alpha_hist
        co_occur = np.zeros((n, n))
        for row in label_hist:
            co_occur += (row[:, None] == row[None, :])
        co_occur /= len(label_hist)
        self._co_matrix_ = co_occur

        Kf = self.labels_.max() + 1
        means = np.empty((Kf, d))
        for k in range(Kf):
            members = X2[self.labels_ == k]
            means[k] = members.mean(axis=0) if len(members) else m0
        self.cluster_means_ = means if d > 1 else means[:, 0]
        self._X_, self._d_ = X2, d
        return self

    def co_clustering_matrix(self):
        return self._co_matrix_

    def predict(self, X_new):
        X_new = np.atleast_1d(np.asarray(X_new, dtype=float))
        if self._d_ == 1 and X_new.ndim == 1:
            X_new = X_new[:, None]
        out = []
        for x in X_new:
            best_k, best_d = None, np.inf
            for k in range(self.labels_.max() + 1):
                dist = np.sum((self.cluster_means_[k] - x) ** 2) if self._d_ > 1 else \
                    (self.cluster_means_[k] - x[0]) ** 2
                if dist < best_d:
                    best_d, best_k = dist, k
            out.append(best_k)
        return np.array(out)

    def predictive_density(self, x_grid):
        if self._d_ != 1:
            raise NotImplementedError("predictive_density is only available for 1-D data")
        x_grid = np.asarray(x_grid, dtype=float)
        dens = np.zeros_like(x_grid)
        S = min(len(self.label_samples_), 200)
        for s in range(S):
            labels = self.label_samples_[-(s + 1)]
            K = labels.max() + 1
            for k in range(K):
                members = self._X_[labels == k, 0]
                w = len(members) / len(labels)
                mu, sd = members.mean(), max(members.std(), 1e-2)
                dens += w * np.exp(-0.5 * ((x_grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        return dens / S

    def __repr__(self):
        return f"DirichletProcess(alpha={self.alpha}, truncation={self.truncation})"
