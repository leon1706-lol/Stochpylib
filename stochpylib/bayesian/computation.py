"""Posterior approximations: Laplace, expectation propagation, mean-field/full-rank
variational inference (delegating to :mod:`stochpylib.advanced_mcmc.variational`), and
Pareto-smoothed self-normalized importance sampling.

VI here is a thin wrapper, not a reimplementation: per the vault's design-scorecard note
(``bayesian.computation`` was thinner than the rest of the module), ``MFVariational``
delegates straight to ``advanced_mcmc.MeanFieldVI``/``ADVI`` rather than duplicating them.
"""

import numpy as np
from scipy import optimize

from stochpylib.bayesian._common import _as_log_density, _cholesky_psd, _rng
from stochpylib.bayesian._result import PosteriorApproximation

__all__ = ["LaplacePosterior", "EP_Posterior", "MFVariational", "ImportanceSamplingPosterior"]


def LaplacePosterior(target, theta0, *, grad=None, hessian=None, method="BFGS", maxiter=500):
    """Gaussian approximation to the posterior at its mode: mean = MAP estimate, cov =
    the inverse negative Hessian of the log-density there."""
    ld = _as_log_density(target)
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    dim = len(theta0)

    def neg_log(theta):
        return -ld(theta)

    def neg_grad(theta):
        return -ld.grad(theta)

    jac = neg_grad if (grad is not None or ld.has_gradient) else None
    res = optimize.minimize(neg_log, theta0, method=method, jac=jac,
                             options={"maxiter": maxiter})
    theta_hat = res.x
    success = bool(res.success) and np.all(np.isfinite(theta_hat)) and np.isfinite(res.fun)
    if not success:
        res2 = optimize.minimize(neg_log, theta0, method="Nelder-Mead",
                                  options={"maxiter": maxiter * 4, "xatol": 1e-8, "fatol": 1e-8})
        if res2.fun < res.fun:
            res, theta_hat = res2, res2.x

    H = hessian(theta_hat) if hessian is not None else ld.hessian(theta_hat)
    H = np.atleast_2d(np.asarray(H, dtype=float))
    neg_H = -H
    jitter_used = 0.0
    try:
        L = _cholesky_psd(neg_H, jitter=1e-10)
    except np.linalg.LinAlgError:
        neg_H = neg_H + 1e-6 * np.eye(dim)
        jitter_used = 1e-6
        L = _cholesky_psd(neg_H)
    cov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(dim)))
    cov = 0.5 * (cov + cov.T)

    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    log_evidence = float(ld(theta_hat) + dim / 2.0 * np.log(2.0 * np.pi) - 0.5 * logdet)

    return PosteriorApproximation(
        "laplace", theta_hat, cov, log_evidence_=log_evidence,
        extras={"hessian": H, "n_evals": ld.n_evals, "optimizer_success": bool(res.success),
                "jitter": jitter_used, "mode": theta_hat})


def EP_Posterior(prior_mean, prior_cov, A, log_site, *, n_iter=50, damping=0.8, tol=1e-6,
                  quad_points=40):
    """Gaussian expectation propagation for ``p(theta) propto N(theta|m0,V0) *
    prod_i t_i(a_i^T theta)``, with sites ``t_i(f) = exp(log_site(f, i))``.

    ``A``: ``(n, d)`` projection rows. ``log_site(f, i)``: vectorized over an array
    ``f`` of cavity-quadrature nodes, for site index ``i``.
    """
    from stochpylib.numerical_methods import GaussHermite

    m0 = np.atleast_1d(np.asarray(prior_mean, dtype=float))
    V0 = np.atleast_2d(np.asarray(prior_cov, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n, d = A.shape
    gh = GaussHermite(quad_points, kind="probabilists")

    tau = np.zeros(n)
    nu = np.zeros(n)
    V0inv = np.linalg.inv(V0)
    V = V0.copy()
    m = m0.copy()

    def refresh_posterior():
        prec = V0inv + (A.T * tau) @ A
        L = _cholesky_psd(prec)
        Vn = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(d)))
        Vn = 0.5 * (Vn + Vn.T)
        mn = Vn @ (V0inv @ m0 + A.T @ nu)
        return mn, Vn

    def tilted_moments(cav_mean, cav_var, i):
        """(Z0, log Z0, m_hat, var_hat) of the tilted distribution q_{-i}(f) t_i(f)."""
        std = np.sqrt(cav_var)
        f_nodes = cav_mean + std * gh.nodes_
        log_site_vals = np.asarray(log_site(f_nodes, i), dtype=float)
        site_max = np.max(log_site_vals)
        w = np.exp(log_site_vals - site_max) * gh.weights_
        Z0 = np.sum(w)
        if Z0 <= 1e-300:
            return Z0, -np.inf, cav_mean, cav_var
        m_hat = np.sum(w * f_nodes) / Z0
        var_hat = max(np.sum(w * (f_nodes - m_hat) ** 2) / Z0, 1e-10)
        return Z0, float(np.log(Z0) + site_max), m_hat, var_hat

    m, V = refresh_posterior()
    n_iter_run = 0
    for it in range(n_iter):
        max_change = 0.0
        n_iter_run = it + 1
        for i in range(n):
            a = A[i]
            f_mean = float(a @ m)
            f_var = float(a @ V @ a)
            cav_prec = 1.0 / max(f_var, 1e-12) - tau[i]
            if cav_prec <= 1e-10:
                continue
            cav_var = 1.0 / cav_prec
            cav_mean = cav_var * (f_mean / max(f_var, 1e-12) - nu[i])

            Z0, _, m_hat, var_hat = tilted_moments(cav_mean, cav_var, i)
            if Z0 <= 1e-300:
                continue

            new_tau_site = 1.0 / var_hat - cav_prec
            new_nu_site = m_hat / var_hat - cav_mean * cav_prec
            new_tau_site = tau[i] + damping * (new_tau_site - tau[i])
            new_nu_site = nu[i] + damping * (new_nu_site - nu[i])
            max_change = max(max_change, abs(new_tau_site - tau[i]), abs(new_nu_site - nu[i]))
            tau[i], nu[i] = new_tau_site, new_nu_site
            m, V = refresh_posterior()
        if max_change < tol:
            break

    # EP log-evidence: log Z = log[int N(theta|m0,V0) prod_i g~_i(theta) dtheta]
    # + sum_i [log Z_i - log int q_{-i}(f) g~_i(f) df], g~_i(f) = exp(-0.5 tau_i f^2 + nu_i f)
    # (Gaussian-site correction subtracted so the approximate sites aren't double-counted).
    prec = V0inv + (A.T * tau) @ A
    L = _cholesky_psd(prec)
    logdet_prec = 2.0 * np.sum(np.log(np.diag(L)))
    _, logdet_V0inv = np.linalg.slogdet(V0inv)
    gauss_term = -0.5 * logdet_prec + 0.5 * logdet_V0inv + 0.5 * m @ np.linalg.solve(V, m) \
        - 0.5 * m0 @ V0inv @ m0
    site_correction = 0.0
    for i in range(n):
        a = A[i]
        f_mean = float(a @ m)
        f_var = float(a @ V @ a)
        cav_prec = 1.0 / max(f_var, 1e-12) - tau[i]
        cav_var = 1.0 / max(cav_prec, 1e-12)
        cav_mean = cav_var * (f_mean / max(f_var, 1e-12) - nu[i])
        _, logZ_i, m_hat, var_hat = tilted_moments(cav_mean, cav_var, i)
        log_site_norm = 0.5 * np.log(var_hat / cav_var) + 0.5 * (m_hat ** 2 / var_hat - cav_mean ** 2 / cav_var)
        site_correction += logZ_i - log_site_norm
    log_evidence = float(gauss_term + site_correction)

    return PosteriorApproximation(
        "ep", m, V, log_evidence_=log_evidence,
        extras={"site_tau": tau, "site_nu": nu, "n_iter_": n_iter_run})


def MFVariational(target, dim, *, rank="mean-field", lower=None, upper=None, n_iter=2000,
                   n_mc=10, lr=0.05, init_mean=None, random_state=None):
    """Gaussian variational posterior via ``advanced_mcmc.MeanFieldVI``/``ADVI``."""
    from stochpylib.advanced_mcmc import MeanFieldVI, ADVI

    ld = _as_log_density(target)
    log_prob = ld.log_prob if hasattr(ld, "log_prob") else ld
    grad_log_prob = ld.grad_log_prob if ld.has_gradient else None

    if rank == "full" or lower is not None or upper is not None:
        engine = ADVI(log_prob, dim, grad_log_prob=grad_log_prob, rank=rank, lower=lower,
                       upper=upper, n_iter=n_iter, n_mc=n_mc, lr=lr, init_mean=init_mean)
        engine.fit(random_state=random_state)
        mean_ = engine.mean_
        if rank == "full" and getattr(engine, "scale_tril_", None) is not None:
            cov_ = engine.scale_tril_ @ engine.scale_tril_.T
        else:
            cov_ = np.diag(engine.std_ ** 2)
    else:
        engine = MeanFieldVI(log_prob, dim, grad_log_prob=grad_log_prob, n_iter=n_iter,
                              n_mc=n_mc, lr=lr, init_mean=init_mean)
        engine.fit(random_state=random_state)
        mean_ = engine.mean_
        cov_ = np.diag(engine.std_ ** 2)

    elbo_hist = engine.elbo_history_
    return PosteriorApproximation(
        "vi", mean_, cov_, log_evidence_=float(elbo_hist[-1]),
        n_iter_=engine.n_iter_, converged_=engine.converged_,
        extras={"engine": engine, "elbo_history": elbo_hist,
                "evidence_is_lower_bound": True})


def ImportanceSamplingPosterior(target, proposal=None, n=4000, *, theta0=None, df=5,
                                 scale=1.5, smooth="psis", random_state=None):
    """Self-normalized importance sampling posterior. Default proposal: a Laplace fit
    widened into a multivariate-t (``df``, ``scale``x the Laplace covariance)."""
    from stochpylib.bayesian._common import _psis
    from stochpylib.distributions import MultivariateT, Normal

    ld = _as_log_density(target)
    rng = _rng(random_state)

    if proposal is None:
        t0 = theta0 if theta0 is not None else np.zeros(1)
        lap = LaplacePosterior(target, t0)
        dim = lap.dim
        shape = scale ** 2 * lap.cov_
        if dim == 1:
            prop_dist = Normal(float(lap.mean_[0]), float(np.sqrt(shape[0, 0])))

            def log_q(theta):
                return float(np.log(np.clip(prop_dist.pdf(theta[0]), 1e-300, None)))

            def sample_q(m, rng2):
                return prop_dist.rvs(m, random_state=rng2)[:, None]
        else:
            prop_dist = MultivariateT(df, lap.mean_, shape)

            def log_q(theta):
                return float(np.log(max(prop_dist.pdf(theta), 1e-300)))

            def sample_q(m, rng2):
                return prop_dist.rvs(m, random_state=rng2)
    else:
        if hasattr(proposal, "sample") and hasattr(proposal, "logpdf"):
            dim = proposal.dim

            def sample_q(m, rng2):
                return proposal.sample(m, random_state=rng2)

            def log_q(theta):
                return proposal.logpdf(theta)
        else:
            sampler, logpdf_fn = proposal
            dim = np.atleast_1d(np.asarray(theta0)).size if theta0 is not None else 1
            sample_q, log_q = sampler, logpdf_fn

    thetas = sample_q(n, rng)
    thetas = np.atleast_2d(thetas)
    log_p = np.array([ld(th) for th in thetas])
    log_q_vals = np.array([log_q(th) for th in thetas])
    log_w = log_p - log_q_vals

    log_evidence = float(np.log(np.sum(np.exp(log_w - log_w.max()))) + log_w.max() - np.log(n))

    if smooth == "psis":
        log_w_smooth, k = _psis(log_w)
    else:
        log_w_smooth, k = log_w - np.log(np.sum(np.exp(log_w - log_w.max()))) - log_w.max(), None
    w = np.exp(log_w_smooth)
    w = w / w.sum()
    mean = np.sum(thetas * w[:, None], axis=0)
    diff = thetas - mean
    cov = (diff * w[:, None]).T @ diff
    ess = float(1.0 / np.sum(w ** 2))

    return PosteriorApproximation(
        "importance", mean, cov, log_evidence_=log_evidence, samples_=thetas,
        weights_=w, extras={"ess": ess, "pareto_k": k, "n": n})
