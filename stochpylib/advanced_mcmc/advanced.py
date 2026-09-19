"""Population/sequential and transdimensional MCMC: replica exchange and parallel
tempering (Geyer 1991, Vousden, Farr & Mandel 2016), adaptive-tempering sequential Monte
Carlo (Del Moral, Doucet & Jasra 2006), particle marginal Metropolis-Hastings (Andrieu,
Doucet & Holenstein 2010), and reversible-jump / product-space transdimensional samplers
(Green 1995, Carlin & Chib 1995).
"""

import numpy as np

from stochpylib.advanced_mcmc._base import LogDensity
from stochpylib.advanced_mcmc._common import (
    _cholesky_psd,
    _log_accept,
    _logsumexp,
    _rng,
    _robbins_monro_scale,
    _systematic_resample,
)

__all__ = ["ParallelTempering", "ParticleMCMC", "ReplicaExchange", "ReversibleJumpMCMC",
           "SequentialMonteCarlo", "TransdimensionalMCMC"]


class ReplicaExchange:
    """General replica-exchange engine: K replicas each with their own log-density,
    random-walk Metropolis moves within each replica, and periodic state swaps between
    replicas. Replica 0 is the one of interest; its draws are returned in ``chains_``
    (shape ``(1, n_samples, dim)``) so it composes with the diagnostics/summary API.

    ``log_probs``: a list of ``K`` callables, or a single ``callable(theta, k)`` together
    with ``n_replicas=K``. ``swap_scheme``: ``"adjacent"`` (one random neighbouring pair
    per swap round) or ``"even-odd"`` (all even pairs, then all odd pairs, alternating).
    """

    def __init__(self, log_probs, n_samples=1000, n_warmup=None, thin=1,
                 proposal_scale=1.0, swap_interval=1, swap_scheme="adjacent",
                 adapt_scale=True, target_accept=0.234, n_replicas=None):
        if callable(log_probs) and n_replicas is not None:
            k = int(n_replicas)
            self.log_probs = [(lambda theta, kk=kk: log_probs(theta, kk)) for kk in range(k)]
        else:
            self.log_probs = list(log_probs)
        self.n_replicas = len(self.log_probs)
        if self.n_replicas < 2:
            raise ValueError("need at least 2 replicas")
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_samples // 2) if n_warmup is None else int(n_warmup)
        self.thin = int(thin)
        self.proposal_scale = proposal_scale
        self.swap_interval = int(swap_interval)
        if swap_scheme not in ("adjacent", "even-odd"):
            raise ValueError("swap_scheme must be 'adjacent' or 'even-odd'")
        self.swap_scheme = swap_scheme
        self.adapt_scale = bool(adapt_scale)
        self.target_accept = float(target_accept)
        self.n_chains = 1

    def _init_thetas(self, theta_init, dim):
        arr = np.atleast_1d(np.asarray(theta_init, dtype=float))
        if arr.ndim == 1:
            return np.tile(arr, (self.n_replicas, 1))
        if arr.shape[0] == self.n_replicas:
            return arr
        raise ValueError("theta_init must be (dim,) or (n_replicas, dim)")

    def sample(self, theta_init, random_state=None):
        rng = _rng(random_state)
        dim = np.atleast_1d(np.asarray(theta_init, dtype=float)).shape[-1]
        thetas = self._init_thetas(theta_init, dim)
        targets = [LogDensity(lp) for lp in self.log_probs]
        logps = np.array([targets[k](thetas[k]) for k in range(self.n_replicas)])
        for k in range(self.n_replicas):
            if not np.isfinite(logps[k]):
                raise ValueError(f"theta_init has zero density under replica {k}")
        log_scales = np.full(self.n_replicas, np.log(self.proposal_scale) if np.ndim(self.proposal_scale) == 0
                             else 0.0)
        scale_vec = np.broadcast_to(np.asarray(self.proposal_scale, dtype=float), (dim,)).copy() \
            if np.ndim(self.proposal_scale) == 0 else np.asarray(self.proposal_scale, dtype=float)
        k_counts = np.zeros(self.n_replicas)
        swap_attempts = np.zeros(self.n_replicas - 1)
        swap_accepts = np.zeros(self.n_replicas - 1)
        n_acc = np.zeros(self.n_replicas)
        n_total_iters = (self.n_warmup + self.n_samples * self.thin)

        chain0 = np.empty((self.n_samples, dim))
        replica_chains = np.empty((self.n_replicas, self.n_samples, dim))
        kept_i = 0
        for it in range(n_total_iters):
            for k in range(self.n_replicas):
                step = np.exp(log_scales[k]) * scale_vec * rng.standard_normal(dim)
                new = thetas[k] + step
                logp_new = targets[k](new)
                log_alpha = logp_new - logps[k]
                accept_prob = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
                if _log_accept(np.log(rng.uniform()), log_alpha):
                    thetas[k] = new
                    logps[k] = logp_new
                    n_acc[k] += 1
                if it < self.n_warmup and self.adapt_scale:
                    k_counts[k] += 1
                    log_scales[k] = _robbins_monro_scale(log_scales[k], accept_prob,
                                                         self.target_accept, k_counts[k])

            if (it + 1) % self.swap_interval == 0:
                pairs = self._swap_pairs(rng)
                for (a, b) in pairs:
                    log_alpha_swap = (targets[a](thetas[b]) + targets[b](thetas[a])
                                      - targets[a](thetas[a]) - targets[b](thetas[b]))
                    swap_attempts[min(a, b)] += 1
                    if _log_accept(np.log(rng.uniform()), log_alpha_swap):
                        thetas[[a, b]] = thetas[[b, a]]
                        logps[a], logps[b] = targets[a](thetas[a]), targets[b](thetas[b])
                        swap_accepts[min(a, b)] += 1

            if it >= self.n_warmup and (it - self.n_warmup + 1) % self.thin == 0:
                replica_chains[:, kept_i, :] = thetas
                chain0[kept_i] = thetas[0]
                kept_i += 1

        self.chains_ = chain0[None, :, :]
        self.replica_chains_ = replica_chains
        self.log_probs_ = None
        self.acceptance_rates_ = n_acc / n_total_iters
        self.acceptance_rate_ = float(self.acceptance_rates_[0])
        self.swap_acceptance_ = np.divide(swap_accepts, swap_attempts,
                                          out=np.zeros_like(swap_accepts),
                                          where=swap_attempts > 0)
        self.dim_ = dim
        return self

    def _swap_pairs(self, rng):
        k = self.n_replicas
        if self.swap_scheme == "adjacent":
            a = int(rng.integers(0, k - 1))
            return [(a, a + 1)]
        # even-odd: alternate all even-indexed pairs and all odd-indexed pairs
        if not hasattr(self, "_eo_toggle"):
            self._eo_toggle = 0
        start = 0 if self._eo_toggle == 0 else 1
        self._eo_toggle = 1 - self._eo_toggle
        return [(i, i + 1) for i in range(start, k - 1, 2)]

    def get_chains(self):
        if not hasattr(self, "chains_"):
            raise RuntimeError("sample() must be called first")
        return self.chains_

    def get_samples(self):
        return self.get_chains().reshape(-1, self.dim_)

    def summary(self):
        from stochpylib.advanced_mcmc.diagnostics import TraceAnalysis
        return TraceAnalysis(self.get_chains()).summary()

    def __repr__(self):
        return f"{self.__class__.__name__}(n_replicas={self.n_replicas}, n_samples={self.n_samples})"


class ParallelTempering(ReplicaExchange):
    """Parallel tempering: a geometric (or user-supplied) temperature ladder applied to a
    single ``log_prob``, i.e. replica ``k`` targets ``beta_k * log_prob(theta)`` with
    ``beta_0 = 1 > beta_1 > ... > beta_{K-1}``. With ``adapt_ladder`` (default), a short
    pilot phase (``pilot_rounds`` rounds of ``pilot_iters`` iterations each) adjusts the
    gaps between adjacent inverse temperatures toward equal swap acceptance (Vousden,
    Farr & Mandel 2016) before the ladder is frozen and the real run starts.
    """

    def __init__(self, log_prob, temperatures=None, n_temps=5, max_temp=30.0,
                 adapt_ladder=True, pilot_rounds=5, pilot_iters=200, **kwargs):
        if temperatures is not None:
            temps = np.asarray(temperatures, dtype=float)
        else:
            k = int(n_temps)
            temps = np.geomspace(1.0, max_temp, k)
        if temps[0] != 1.0:
            raise ValueError("temperatures[0] must be 1.0 (the target of interest)")
        self.temperatures_ = temps
        self.betas_ = 1.0 / temps
        self.adapt_ladder = bool(adapt_ladder)
        self.pilot_rounds = int(pilot_rounds)
        self.pilot_iters = int(pilot_iters)
        self._base_log_prob = log_prob
        self._kwargs = dict(kwargs)
        super().__init__(self._make_tempered(self.betas_), n_replicas=len(temps), **kwargs)

    def _make_tempered(self, betas):
        log_prob = self._base_log_prob
        return lambda theta, k, betas=betas: betas[k] * log_prob(theta)

    def sample(self, theta_init, random_state=None):
        rng = _rng(random_state)
        if self.adapt_ladder:
            self._adapt_ladder_pilot(theta_init, rng)
            self.log_probs = [(lambda theta, kk=kk: self._make_tempered(self.betas_)(theta, kk))
                              for kk in range(len(self.temperatures_))]
        return super().sample(theta_init, random_state=rng)

    def _adapt_ladder_pilot(self, theta_init, rng):
        temps = self.temperatures_.copy()
        pilot_kwargs = {k: v for k, v in self._kwargs.items()
                        if k not in ("n_samples", "n_warmup", "thin")}
        for round_idx in range(self.pilot_rounds):
            betas = 1.0 / temps
            pilot = ReplicaExchange(self._make_tempered(betas), n_replicas=len(temps),
                                    n_samples=1, n_warmup=self.pilot_iters, thin=1,
                                    **pilot_kwargs)
            pilot.sample(theta_init, random_state=rng)
            A = pilot.swap_acceptance_
            kappa = 0.5 / (1.0 + round_idx)
            for k in range(1, len(temps) - 1):
                gap = np.log(max(temps[k] - temps[k - 1], 1e-6))
                gap = gap + kappa * (A[k - 1] - A[k])
                temps[k] = temps[k - 1] + np.exp(gap)
            temps = np.sort(temps)
            temps[0] = 1.0
        self.temperatures_ = temps
        self.betas_ = 1.0 / temps


class SequentialMonteCarlo:
    """Adaptive-tempering sequential Monte Carlo (Del Moral, Doucet & Jasra 2006):
    particles move from the prior to the posterior through a bisection-chosen tempering
    schedule, with systematic resampling and random-walk-Metropolis rejuvenation at each
    stage. ``log_evidence_`` estimates the marginal likelihood (model evidence).
    """

    def __init__(self, log_prior, log_likelihood, prior_sampler, n_particles=1000,
                 ess_threshold=0.5, n_mcmc=3, proposal_scale=1.0, max_stages=200):
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.prior_sampler = prior_sampler
        self.n_particles = int(n_particles)
        self.ess_threshold = float(ess_threshold)
        self.n_mcmc = int(n_mcmc)
        self.proposal_scale = float(proposal_scale)
        self.max_stages = int(max_stages)

    def sample(self, random_state=None):
        rng = _rng(random_state)
        n = self.n_particles
        x = np.atleast_2d(self.prior_sampler(n, rng))
        dim = x.shape[1]
        ll = np.array([self.log_likelihood(xi) for xi in x])
        lp = np.array([self.log_prior(xi) for xi in x])
        beta = 0.0
        betas = [0.0]
        logZ = 0.0
        acceptances = []
        stage = 0
        while beta < 1.0 - 1e-12:
            new_beta = self._next_beta(ll, beta, n)
            log_w = (new_beta - beta) * ll
            logZ += float(_logsumexp(log_w)) - np.log(n)
            m = log_w.max()
            w = np.exp(log_w - m)
            w = w / w.sum()
            idx = _systematic_resample(w, rng)
            x, ll, lp = x[idx], ll[idx], lp[idx]

            cov = np.atleast_2d(np.cov(x.T, ddof=1)) if dim > 1 else np.array([[x[:, 0].var(ddof=1) + 1e-12]])
            cov = (2.38 ** 2 / dim) * cov * self.proposal_scale + 1e-10 * np.eye(dim)
            L = _cholesky_psd(cov)
            n_acc = 0
            for _ in range(self.n_mcmc):
                prop = x + (L @ rng.standard_normal((dim, n))).T
                lp_new = np.array([self.log_prior(xi) for xi in prop])
                ll_new = np.array([self.log_likelihood(xi) for xi in prop])
                log_alpha = (lp_new + new_beta * ll_new) - (lp + new_beta * ll)
                u = np.log(rng.uniform(size=n))
                accept = (u < log_alpha) & np.isfinite(log_alpha)
                x = np.where(accept[:, None], prop, x)
                lp = np.where(accept, lp_new, lp)
                ll = np.where(accept, ll_new, ll)
                n_acc += accept.sum()
            acceptances.append(n_acc / (n * self.n_mcmc))
            beta = new_beta
            betas.append(beta)
            stage += 1
            if stage > self.max_stages:
                raise RuntimeError("SMC exceeded max_stages without reaching beta=1")

        self.particles_ = x
        self.log_evidence_ = float(logZ)
        self.betas_ = np.array(betas)
        self.acceptance_rates_ = np.array(acceptances)
        self.n_stages_ = stage
        self.dim_ = dim
        self.chains_ = x[None, :, :]
        return self

    def _next_beta(self, ll, beta, n):
        def ess_at(new_beta):
            log_w = (new_beta - beta) * ll
            m = log_w.max()
            w = np.exp(log_w - m)
            w = w / w.sum()
            return 1.0 / np.sum(w ** 2)

        if ess_at(1.0) >= self.ess_threshold * n:
            return 1.0
        lo, hi = beta, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ess_at(mid) >= self.ess_threshold * n:
                lo = mid
            else:
                hi = mid
        return max(lo, beta + 1e-8)

    def get_samples(self):
        if not hasattr(self, "particles_"):
            raise RuntimeError("sample() must be called first")
        return self.particles_

    def get_chains(self):
        if not hasattr(self, "chains_"):
            raise RuntimeError("sample() must be called first")
        return self.chains_

    def summary(self):
        from stochpylib.advanced_mcmc.diagnostics import TraceAnalysis
        return TraceAnalysis(self.chains_).summary()

    def __repr__(self):
        return f"SequentialMonteCarlo(n_particles={self.n_particles})"


class ParticleMCMC:
    """Particle marginal Metropolis-Hastings (Andrieu, Doucet & Holenstein 2010) for
    Bayesian inference in state-space models: parameters ``theta`` are updated by a
    random-walk MH step whose acceptance ratio uses an unbiased particle-filter estimate
    of the marginal likelihood (via :class:`stochpylib.timeseries.ParticleFilter`), which
    makes the resulting chain exact for ``theta`` despite the likelihood being estimated,
    not computed in closed form.

    ``initial_sampler(theta, rng) -> initial particle array``,
    ``transition_sampler(particles, theta, rng) -> new particles``,
    ``observation_logpdf(particles, y, theta) -> log p(y | x)`` (vectorized over particles).
    """

    def __init__(self, observations, log_prior, initial_sampler, transition_sampler,
                 observation_logpdf, n_particles=200, n_samples=1000, n_warmup=None,
                 n_chains=1, thin=1, proposal_scale=0.1, proposal_cov=None,
                 adapt_scale=True, target_accept=0.234):
        self.observations = np.asarray(observations, dtype=float)
        self.log_prior = log_prior
        self.initial_sampler = initial_sampler
        self.transition_sampler = transition_sampler
        self.observation_logpdf = observation_logpdf
        self.n_particles = int(n_particles)
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_samples // 2) if n_warmup is None else int(n_warmup)
        self.n_chains = int(n_chains)
        self.thin = int(thin)
        self.proposal_scale = proposal_scale
        self.proposal_cov = proposal_cov
        self.adapt_scale = bool(adapt_scale)
        self.target_accept = float(target_accept)

    def _loglik_hat(self, theta, rng):
        from stochpylib.timeseries import ParticleFilter

        pf = ParticleFilter(
            n_particles=self.n_particles,
            transition_sampler=lambda p, r, th=theta: self.transition_sampler(p, th, r),
            observation_logpdf=lambda p, y, th=theta: self.observation_logpdf(p, y, th),
            initial_sampler=lambda r, th=theta: self.initial_sampler(th, r),
            random_state=rng,
        )
        pf.fit(self.observations)
        return float(pf.loglik_)

    def sample(self, theta_init, random_state=None):
        from stochpylib.advanced_mcmc._common import _broadcast_init, _spawn_rngs

        rng = _rng(random_state)
        inits = _broadcast_init(theta_init, self.n_chains)
        dim = inits.shape[1]
        rngs = _spawn_rngs(rng, self.n_chains)
        chains = np.empty((self.n_chains, self.n_samples, dim))
        loglik_ests = np.empty((self.n_chains, self.n_samples))
        acc = np.zeros(self.n_chains)
        L = _cholesky_psd(np.asarray(self.proposal_cov, dtype=float)) if self.proposal_cov is not None else None
        scale = np.asarray(self.proposal_scale, dtype=float)
        scale_vec = np.broadcast_to(scale, (dim,)).copy() if scale.ndim == 0 else scale

        for c in range(self.n_chains):
            r = rngs[c]
            theta = inits[c]
            loglik_hat = self._loglik_hat(theta, r)
            logp = float(self.log_prior(theta)) + loglik_hat
            log_scale = 0.0
            k = 0
            n_total = self.n_samples * self.thin
            kept = 0
            n_acc = 0.0
            for i in range(self.n_warmup + n_total):
                z = r.standard_normal(dim)
                step = L @ z if L is not None else z
                new = theta + np.exp(log_scale) * scale_vec * step
                loglik_new = self._loglik_hat(new, r)
                logp_new = float(self.log_prior(new)) + loglik_new
                log_alpha = logp_new - logp
                accept_prob = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
                accepted = _log_accept(np.log(r.uniform()), log_alpha)
                if accepted:
                    theta, logp, loglik_hat = new, logp_new, loglik_new
                if i < self.n_warmup:
                    if self.adapt_scale:
                        k += 1
                        log_scale = _robbins_monro_scale(log_scale, accept_prob, self.target_accept, k)
                else:
                    n_acc += float(accepted)
                    if (i - self.n_warmup + 1) % self.thin == 0:
                        chains[c, kept] = theta
                        loglik_ests[c, kept] = loglik_hat
                        kept += 1
            acc[c] = n_acc / n_total

        self.chains_ = chains
        self.loglik_estimates_ = loglik_ests
        self.acceptance_rates_ = acc
        self.acceptance_rate_ = float(acc.mean())
        self.dim_ = dim
        return self

    def get_chains(self):
        if not hasattr(self, "chains_"):
            raise RuntimeError("sample() must be called first")
        return self.chains_

    def get_samples(self):
        return self.get_chains().reshape(-1, self.dim_)

    def summary(self):
        from stochpylib.advanced_mcmc.diagnostics import TraceAnalysis
        return TraceAnalysis(self.get_chains()).summary()

    def __repr__(self):
        return f"ParticleMCMC(n_particles={self.n_particles}, n_samples={self.n_samples})"


def _default_jump(k, theta, k_new, dims, jump_scale, rng):
    """Default dimension-matching move: pad with Gaussian auxiliaries when growing, drop
    trailing coordinates when shrinking; identity when the dimension is unchanged."""
    d_old, d_new = dims[k], dims[k_new]
    if d_new == d_old:
        return theta.copy(), 0.0
    if d_new > d_old:
        extra = rng.standard_normal(d_new - d_old) * jump_scale
        theta_new = np.concatenate([theta, extra])
        log_ratio = -np.sum(-0.5 * (extra / jump_scale) ** 2 - np.log(jump_scale) - 0.5 * np.log(2 * np.pi))
        return theta_new, log_ratio
    dropped = theta[d_new:]
    theta_new = theta[:d_new].copy()
    log_ratio = np.sum(-0.5 * (dropped / jump_scale) ** 2 - np.log(jump_scale) - 0.5 * np.log(2 * np.pi))
    return theta_new, log_ratio


class ReversibleJumpMCMC:
    """Reversible-jump MCMC (Green 1995) over a finite set of candidate models.

    ``log_posteriors``: dict ``model_key -> callable(theta) -> log(prior * likelihood)``
    (up to a constant shared across models). ``dims``: dict ``model_key -> dimension``.
    ``jump(k, theta, k_new, rng) -> (theta_new, log_ratio)`` where ``log_ratio`` is
    ``log q(reverse) - log q(forward) + log|Jacobian|``; defaults to the Gaussian-
    auxiliary padding/dropping move above, which is a valid (if generic) choice whenever
    models are nested by dropping trailing coordinates.
    """

    def __init__(self, log_posteriors, dims, log_model_prior=None, jump=None,
                 n_samples=1000, n_warmup=None, within_scale=0.5, jump_scale=1.0,
                 p_jump=0.3, adapt_scale=True):
        self.log_posteriors = dict(log_posteriors)
        self.dims = dict(dims)
        self.models = list(self.log_posteriors)
        if log_model_prior is None:
            log_model_prior = {m: -np.log(len(self.models)) for m in self.models}
        self.log_model_prior = dict(log_model_prior)
        self.jump_fn = jump
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_samples // 2) if n_warmup is None else int(n_warmup)
        self.within_scale = float(within_scale)
        self.jump_scale = float(jump_scale)
        self.p_jump = float(p_jump)
        self.adapt_scale = bool(adapt_scale)
        self.n_chains = 1

    def _jump(self, k, theta, k_new, rng):
        if self.jump_fn is not None:
            return self.jump_fn(k, theta, k_new, rng)
        return _default_jump(k, theta, k_new, self.dims, self.jump_scale, rng)

    def sample(self, model_init, theta_init, random_state=None):
        rng = _rng(random_state)
        k = model_init
        theta = np.asarray(theta_init, dtype=float)
        logp = self.log_posteriors[k](theta) + self.log_model_prior[k]
        n_models = len(self.models)
        log_scales = {m: np.log(self.within_scale) for m in self.models}
        k_counts = {m: 0 for m in self.models}
        model_chain = np.empty(self.n_samples, dtype=object)
        samples_by_model = {m: [] for m in self.models}
        n_acc_within = 0.0
        n_within = 0
        n_acc_jump = 0.0
        n_jump = 0
        n_total = self.n_warmup + self.n_samples

        for i in range(n_total):
            if rng.uniform() < self.p_jump and n_models > 1:
                others = [m for m in self.models if m != k]
                k_new = others[int(rng.integers(len(others)))]
                theta_new, log_ratio = self._jump(k, theta, k_new, rng)
                logp_new = self.log_posteriors[k_new](theta_new) + self.log_model_prior[k_new]
                logp_old_model = self.log_posteriors[k](theta) + self.log_model_prior[k]
                log_alpha = (logp_new - logp_old_model) + log_ratio
                if i >= self.n_warmup:
                    n_jump += 1
                if _log_accept(np.log(rng.uniform()), log_alpha):
                    k, theta, logp = k_new, theta_new, logp_new
                    if i >= self.n_warmup:
                        n_acc_jump += 1
            else:
                dim = self.dims[k]
                step = np.exp(log_scales[k]) * rng.standard_normal(dim)
                theta_new = theta + step
                logp_new = self.log_posteriors[k](theta_new) + self.log_model_prior[k]
                log_alpha = logp_new - logp
                accept_prob = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
                accepted = _log_accept(np.log(rng.uniform()), log_alpha)
                if accepted:
                    theta, logp = theta_new, logp_new
                if i < self.n_warmup and self.adapt_scale:
                    k_counts[k] += 1
                    log_scales[k] = _robbins_monro_scale(log_scales[k], accept_prob, 0.234, k_counts[k])
                if i >= self.n_warmup:
                    n_within += 1
                    n_acc_within += float(accepted)

            if i >= self.n_warmup:
                idx = i - self.n_warmup
                model_chain[idx] = k
                samples_by_model[k].append(theta.copy())

        self.model_chain_ = model_chain
        counts = {m: 0 for m in self.models}
        for m in model_chain:
            counts[m] += 1
        self.model_probabilities_ = {m: counts[m] / self.n_samples for m in self.models}
        self.samples_by_model_ = {m: (np.array(v) if v else np.empty((0, self.dims[m])))
                                  for m, v in samples_by_model.items()}
        self.acceptance_rate_ = n_acc_within / n_within if n_within > 0 else float("nan")
        self.jump_acceptance_ = n_acc_jump / n_jump if n_jump > 0 else float("nan")
        return self

    def get_samples(self, model):
        if not hasattr(self, "samples_by_model_"):
            raise RuntimeError("sample() must be called first")
        return self.samples_by_model_[model]

    def __repr__(self):
        return f"ReversibleJumpMCMC(models={self.models}, n_samples={self.n_samples})"


class TransdimensionalMCMC:
    """Carlin & Chib (1995) product-space sampler: every model's parameters are updated
    at every iteration (drawn from a fitted Gaussian pseudo-prior when not the current
    model), and the model indicator is resampled from the resulting product-space
    conditional -- an alternative to reversible jump that needs no dimension-matching
    move, at the cost of fitting one pseudo-prior per model from a pilot run.
    """

    def __init__(self, log_posteriors, dims, log_model_prior=None, n_samples=1000,
                 n_warmup=None, pilot_samples=500, within_scale=0.5, adapt_scale=True):
        self.log_posteriors = dict(log_posteriors)
        self.dims = dict(dims)
        self.models = list(self.log_posteriors)
        if log_model_prior is None:
            log_model_prior = {m: -np.log(len(self.models)) for m in self.models}
        self.log_model_prior = dict(log_model_prior)
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_samples // 2) if n_warmup is None else int(n_warmup)
        self.pilot_samples = int(pilot_samples)
        self.within_scale = float(within_scale)
        self.adapt_scale = bool(adapt_scale)
        self.n_chains = 1

    def _fit_pseudo_prior(self, m, theta0, rng):
        from stochpylib.advanced_mcmc.standard import MetropolisHastings

        sampler = MetropolisHastings(self.log_posteriors[m], n_samples=self.pilot_samples,
                                     n_warmup=self.pilot_samples // 2,
                                     proposal_scale=self.within_scale)
        sampler.sample(theta0, random_state=rng)
        draws = sampler.get_samples()
        mean = draws.mean(axis=0)
        cov = np.atleast_2d(np.cov(draws.T, ddof=1)) + 1e-6 * np.eye(self.dims[m])
        return mean, cov, _cholesky_psd(cov)

    def _pseudo_logpdf(self, x, mean, cov_inv, logdet):
        diff = x - mean
        return float(-0.5 * diff @ cov_inv @ diff - 0.5 * logdet - 0.5 * len(x) * np.log(2 * np.pi))

    def sample(self, theta_init, random_state=None):
        rng = _rng(random_state)
        pseudo = {}
        for m in self.models:
            mean, cov, L = self._fit_pseudo_prior(m, theta_init[m], rng)
            cov_inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            pseudo[m] = {"mean": mean, "cov": cov, "L": L, "cov_inv": cov_inv, "logdet": logdet}
        self.pseudo_priors_ = pseudo

        thetas = {m: np.asarray(theta_init[m], dtype=float).copy() for m in self.models}
        k = self.models[0]
        log_scales = {m: np.log(self.within_scale) for m in self.models}
        k_counts = {m: 0 for m in self.models}
        n_acc, n_total_within = 0.0, 0
        model_chain = np.empty(self.n_samples, dtype=object)
        samples_by_model = {m: [] for m in self.models}
        n_total = self.n_warmup + self.n_samples

        for i in range(n_total):
            # update the current model's parameters with one RW-MH step
            dim = self.dims[k]
            step = np.exp(log_scales[k]) * rng.standard_normal(dim)
            theta_new = thetas[k] + step
            logp_old = self.log_posteriors[k](thetas[k])
            logp_new = self.log_posteriors[k](theta_new)
            log_alpha = logp_new - logp_old
            accept_prob = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
            accepted = _log_accept(np.log(rng.uniform()), log_alpha)
            if accepted:
                thetas[k] = theta_new
            if i < self.n_warmup and self.adapt_scale:
                k_counts[k] += 1
                log_scales[k] = _robbins_monro_scale(log_scales[k], accept_prob, 0.234, k_counts[k])
            else:
                n_total_within += 1
                n_acc += float(accepted)

            # draw pseudo-priors for every other model
            for m in self.models:
                if m != k:
                    p = pseudo[m]
                    thetas[m] = p["mean"] + p["L"] @ rng.standard_normal(self.dims[m])

            # resample the model indicator from the product-space conditional
            log_w = {}
            for m in self.models:
                lw = self.log_posteriors[m](thetas[m]) + self.log_model_prior[m]
                for other in self.models:
                    if other != m:
                        p = pseudo[other]
                        lw += self._pseudo_logpdf(thetas[other], p["mean"], p["cov_inv"], p["logdet"])
                log_w[m] = lw
            keys = list(log_w)
            vals = np.array([log_w[kk] for kk in keys])
            vals = vals - vals.max()
            probs = np.exp(vals)
            probs = probs / probs.sum()
            k = keys[int(rng.choice(len(keys), p=probs))]

            if i >= self.n_warmup:
                idx = i - self.n_warmup
                model_chain[idx] = k
                samples_by_model[k].append(thetas[k].copy())

        self.model_chain_ = model_chain
        counts = {m: 0 for m in self.models}
        for m in model_chain:
            counts[m] += 1
        self.model_probabilities_ = {m: counts[m] / self.n_samples for m in self.models}
        self.samples_by_model_ = {m: (np.array(v) if v else np.empty((0, self.dims[m])))
                                  for m, v in samples_by_model.items()}
        self.acceptance_rate_ = n_acc / n_total_within if n_total_within > 0 else float("nan")
        return self

    def get_samples(self, model):
        if not hasattr(self, "samples_by_model_"):
            raise RuntimeError("sample() must be called first")
        return self.samples_by_model_[model]

    def __repr__(self):
        return f"TransdimensionalMCMC(models={self.models}, n_samples={self.n_samples})"
