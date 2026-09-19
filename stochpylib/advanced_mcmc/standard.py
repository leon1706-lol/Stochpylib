"""Standard (derivative-free) MCMC: random-walk / independence / Gibbs Metropolis-Hastings
and its adaptive variants (Haario et al. 2001, Vihola 2012).
"""

import numpy as np

from stochpylib.advanced_mcmc._base import MCMCSampler
from stochpylib.advanced_mcmc._common import _cholesky_psd, _log_accept, _robbins_monro_scale, _Welford

__all__ = ["AdaptiveMetropolis", "GibbsSampler", "IndependenceSampler", "MetropolisHastings",
           "RobustAdaptiveMetropolis"]


class MetropolisHastings(MCMCSampler):
    """Random-walk Metropolis-Hastings, or a fully general MH kernel with a custom proposal.

    With no ``proposal``, the proposal is a symmetric Gaussian random walk with covariance
    ``proposal_scale**2 * proposal_cov`` (``proposal_cov`` defaults to the identity;
    ``proposal_scale`` may be a scalar or a ``(dim,)`` vector of per-coordinate scales),
    optionally Robbins-Monro-adapted during warmup toward ``target_accept``. With a custom
    ``proposal(theta, rng) -> theta_new``, pass the matching ``proposal_log_density(to,
    from_) -> log q`` for correctness when the proposal is asymmetric (e.g. a multiplicative
    log-normal step); omitting it silently assumes symmetry, which biases the chain if the
    proposal genuinely is not symmetric.
    """

    def __init__(self, log_prob, n_samples=1000, n_warmup=None, n_chains=1, thin=1,
                 grad_log_prob=None, proposal_scale=1.0, proposal_cov=None, proposal=None,
                 proposal_log_density=None, adapt_scale=True, target_accept=0.234):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.proposal_scale = proposal_scale
        self.proposal_cov = proposal_cov
        self.proposal = proposal
        self.proposal_log_density = proposal_log_density
        self.adapt_scale = bool(adapt_scale) and proposal is None
        self.target_accept = float(target_accept)

    def _start_chain(self, dim, rng):
        scale = np.asarray(self.proposal_scale, dtype=float)
        self._scale_vec = np.broadcast_to(scale, (dim,)).copy() if scale.ndim == 0 else scale
        self._log_scale = 0.0
        self._L = _cholesky_psd(np.asarray(self.proposal_cov, dtype=float)) if self.proposal_cov is not None else None
        self._k = 0

    def _step(self, state, rng, adapt):
        theta, logp = state["theta"], state["logp"]
        dim = len(theta)
        if self.proposal is not None:
            new = np.asarray(self.proposal(theta, rng), dtype=float)
            log_q = 0.0
            if self.proposal_log_density is not None:
                log_q = self.proposal_log_density(theta, new) - self.proposal_log_density(new, theta)
        else:
            z = rng.standard_normal(dim)
            step = self._L @ z if self._L is not None else z
            new = theta + np.exp(self._log_scale) * self._scale_vec * step
            log_q = 0.0
        logp_new = self.target(new)
        log_alpha = logp_new - logp + log_q
        u_log = np.log(rng.uniform())
        accepted = _log_accept(u_log, log_alpha)
        accept_prob = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
        new_state = {"theta": new, "logp": logp_new} if accepted else state
        if adapt and self.adapt_scale:
            self._k += 1
            self._log_scale = _robbins_monro_scale(self._log_scale, accept_prob, self.target_accept, self._k)
        return new_state, float(accepted)

    def _end_warmup(self):
        self.proposal_scale_ = np.exp(self._log_scale) * self._scale_vec


class IndependenceSampler(MCMCSampler):
    """Independence Metropolis-Hastings: proposals are drawn independently of the current
    state from a fixed distribution, corrected by the importance-weight-like MH ratio."""

    def __init__(self, log_prob, proposal_sampler, proposal_log_prob, n_samples=1000,
                 n_warmup=None, n_chains=1, thin=1, grad_log_prob=None):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.proposal_sampler = proposal_sampler
        self.proposal_log_prob = proposal_log_prob

    def _init_state(self, theta, rng):
        logp = self.target(theta)
        return {"theta": theta, "logp": logp, "logq": self.proposal_log_prob(theta)}

    def _step(self, state, rng, adapt):
        new = np.asarray(self.proposal_sampler(rng), dtype=float)
        logq_new = float(self.proposal_log_prob(new))
        logp_new = self.target(new)
        log_alpha = (logp_new - state["logp"]) + (state["logq"] - logq_new)
        accepted = _log_accept(np.log(rng.uniform()), log_alpha)
        if accepted:
            return {"theta": new, "logp": logp_new, "logq": logq_new}, 1.0
        return state, 0.0


class GibbsSampler(MCMCSampler):
    """Systematic- or random-scan Gibbs sampling.

    With ``conditionals`` given (a list of callables ``conditionals[i](theta, rng) ->
    new_block_values`` for each entry of ``blocks``, default one coordinate per block),
    each block is drawn exactly from its full conditional and ``acceptance_rate_`` is 1.
    Without ``conditionals``, ``log_prob`` is required and each coordinate is updated by a
    random-walk Metropolis step (Metropolis-within-Gibbs) with a per-coordinate scale
    adapted toward ``target_accept`` during warmup.
    """

    def __init__(self, log_prob=None, conditionals=None, blocks=None, dim=None,
                 scan="systematic", n_samples=1000, n_warmup=None, n_chains=1, thin=1,
                 grad_log_prob=None, proposal_scale=1.0, target_accept=0.44):
        if conditionals is None and log_prob is None:
            raise ValueError("either conditionals or log_prob must be given")
        if log_prob is None:
            log_prob = lambda theta: 0.0
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.conditionals = conditionals
        self.blocks = blocks
        self.dim = dim
        if scan not in ("systematic", "random"):
            raise ValueError("scan must be 'systematic' or 'random'")
        self.scan = scan
        self.proposal_scale = float(proposal_scale)
        self.target_accept = float(target_accept)
        self._exact = conditionals is not None

    def _start_chain(self, dim, rng):
        self._blocks = self.blocks if self.blocks is not None else [[i] for i in range(dim)]
        self._log_scales = np.full(len(self._blocks), np.log(self.proposal_scale))
        self._k = np.zeros(len(self._blocks))

    def _step(self, state, rng, adapt):
        theta = state["theta"].copy()
        order = np.arange(len(self._blocks))
        if self.scan == "random":
            rng.shuffle(order)
        accepts = []
        for bi in order:
            block = self._blocks[bi]
            if self._exact:
                theta[block] = np.atleast_1d(self.conditionals[bi](theta, rng))
                accepts.append(1.0)
            else:
                old_vals = theta[block].copy()
                logp_old = self.target(theta)
                step = np.exp(self._log_scales[bi]) * rng.standard_normal(len(block))
                theta[block] = old_vals + step
                logp_new = self.target(theta)
                log_alpha = logp_new - logp_old
                accept_prob = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
                if not _log_accept(np.log(rng.uniform()), log_alpha):
                    theta[block] = old_vals
                accepts.append(accept_prob)
                if adapt:
                    self._k[bi] += 1
                    self._log_scales[bi] = _robbins_monro_scale(
                        self._log_scales[bi], accept_prob, self.target_accept, self._k[bi])
        logp = self.target(theta) if not self._exact else 0.0
        return {"theta": theta, "logp": logp}, float(np.mean(accepts))


class AdaptiveMetropolis(MCMCSampler):
    """Haario, Saksman & Tamminen (2001) adaptive Metropolis.

    Proposal covariance is ``initial_scale**2 * I`` for the first ``initial_period``
    iterations, then ``scale_factor * (empirical covariance of past draws + epsilon * I)``
    with ``scale_factor`` defaulting to the standard ``2.38**2 / dim``. Adaptation runs
    only during warmup; the covariance is frozen afterward.
    """

    def __init__(self, log_prob, n_samples=1000, n_warmup=None, n_chains=1, thin=1,
                 grad_log_prob=None, initial_scale=0.1, initial_period=50, epsilon=1e-6,
                 scale_factor=None):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.initial_scale = float(initial_scale)
        self.initial_period = int(initial_period)
        self.epsilon = float(epsilon)
        self.scale_factor = scale_factor

    def _start_chain(self, dim, rng):
        self._welford = _Welford(dim)
        self._k = 0
        self._sd = self.scale_factor if self.scale_factor is not None else 2.38 ** 2 / dim
        self._current_cov = (self.initial_scale ** 2) * np.eye(dim)
        self._dim = dim

    def _step(self, state, rng, adapt):
        theta, logp = state["theta"], state["logp"]
        if self._k < self.initial_period:
            cov_prop = (self.initial_scale ** 2) * np.eye(self._dim)
        else:
            cov_prop = self._sd * (self._welford.cov() + self.epsilon * np.eye(self._dim))
        self._current_cov = cov_prop
        L = _cholesky_psd(cov_prop)
        new = theta + L @ rng.standard_normal(self._dim)
        logp_new = self.target(new)
        log_alpha = logp_new - logp
        accepted = _log_accept(np.log(rng.uniform()), log_alpha)
        new_state = {"theta": new, "logp": logp_new} if accepted else state
        if adapt:
            self._welford.update(new_state["theta"])
            self._k += 1
        return new_state, float(accepted)

    def _end_warmup(self):
        self.proposal_cov_ = self._current_cov


class RobustAdaptiveMetropolis(MCMCSampler):
    """Vihola (2012) robust adaptive Metropolis: the proposal's Cholesky factor is adapted
    directly toward a target acceptance rate (no covariance estimate needed)."""

    def __init__(self, log_prob, n_samples=1000, n_warmup=None, n_chains=1, thin=1,
                 grad_log_prob=None, initial_scale=1.0, target_accept=0.234, gamma=2.0 / 3.0):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.initial_scale = float(initial_scale)
        self.target_accept = float(target_accept)
        self.gamma = float(gamma)

    def _start_chain(self, dim, rng):
        self._S = self.initial_scale * np.eye(dim)
        self._k = 0
        self._dim = dim

    def _step(self, state, rng, adapt):
        theta, logp = state["theta"], state["logp"]
        u = rng.standard_normal(self._dim)
        new = theta + self._S @ u
        logp_new = self.target(new)
        log_alpha = logp_new - logp
        alpha = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
        accepted = _log_accept(np.log(rng.uniform()), log_alpha)
        new_state = {"theta": new, "logp": logp_new} if accepted else state
        if adapt:
            self._k += 1
            eta = min(1.0, self._dim * self._k ** (-self.gamma))
            uu = float(u @ u)
            if uu > 1e-300:
                M = self._S @ (np.eye(self._dim) + eta * (alpha - self.target_accept) *
                               np.outer(u, u) / uu) @ self._S.T
                self._S = _cholesky_psd(M)
        return new_state, float(accepted)

    def _end_warmup(self):
        self.proposal_chol_ = self._S
