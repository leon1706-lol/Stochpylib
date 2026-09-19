"""Gradient-based MCMC: Hamiltonian Monte Carlo and NUTS (Hoffman & Gelman 2014,
Betancourt 2017), MALA and manifold MALA, Riemannian HMC (Girolami & Calderhead 2011),
and NeuTra HMC (Hoffman et al. 2019). Gradients are analytic when ``grad_log_prob`` is
given, else :class:`~stochpylib.advanced_mcmc._base.LogDensity` supplies a central
finite-difference fallback.
"""

import numpy as np

from stochpylib.advanced_mcmc._base import MCMCSampler
from stochpylib.advanced_mcmc._common import _DualAveraging, _Welford, _cholesky_psd, _log_accept

__all__ = ["HamiltonianMonteCarlo", "MALA", "MMALA", "NeutraHMC", "NoUTurnSampler", "RiemannianHMC"]


# --------------------------------------------------------------------------- shared bits

def _kinetic(p, inv_mass):
    if inv_mass.ndim == 1:
        return 0.5 * float(p @ (inv_mass * p))
    return 0.5 * float(p @ (inv_mass @ p))


def _sample_momentum(rng, inv_mass, diag):
    """Draw momentum ``p ~ N(0, M)`` where ``M`` is the mass matrix, given its inverse
    ``inv_mass`` (the parameterization every sampler here adapts and stores)."""
    if diag:
        return rng.standard_normal(len(inv_mass)) * np.sqrt(1.0 / inv_mass)
    L = _cholesky_psd(np.linalg.inv(inv_mass))
    return L @ rng.standard_normal(inv_mass.shape[0])


def _leapfrog(target, theta, p, grad, eps, inv_mass, n_steps, diag):
    theta = theta.copy()
    p = p + 0.5 * eps * grad
    for i in range(n_steps):
        velocity = inv_mass * p if diag else inv_mass @ p
        theta = theta + eps * velocity
        grad = target.grad(theta)
        if i < n_steps - 1:
            p = p + eps * grad
        else:
            p = p + 0.5 * eps * grad
    logp = target(theta)
    return theta, p, grad, logp


def _find_reasonable_step_size(target, theta, grad, inv_mass, rng, diag):
    """Hoffman & Gelman (2014) Algorithm 4: bracket a step size where the one-leapfrog-
    step acceptance probability crosses 0.5, doubling or halving until it does."""
    eps = 1.0
    p = _sample_momentum(rng, inv_mass, diag)
    H0 = -target(theta) + _kinetic(p, inv_mass)
    _, p1, _, logp1 = _leapfrog(target, theta, p, grad, eps, inv_mass, 1, diag)
    H1 = -logp1 + _kinetic(p1, inv_mass)
    diff = H0 - H1
    a = 1.0 if (np.isfinite(diff) and diff > np.log(0.5)) else -1.0
    tries = 0
    while tries < 100:
        _, p1, _, logp1 = _leapfrog(target, theta, p, grad, eps, inv_mass, 1, diag)
        H1 = -logp1 + _kinetic(p1, inv_mass)
        diff = H0 - H1
        if not np.isfinite(diff):
            eps *= 0.5 if a > 0 else 2.0
            tries += 1
            continue
        # a=+1: keep doubling while acceptance prob stays above 0.5; a=-1: keep halving
        # while it stays below 0.5. Stop once it has crossed to the other side.
        if a * diff <= a * np.log(0.5):
            break
        eps = eps * (2.0 ** a)
        tries += 1
    return max(eps, 1e-6)


class _WindowedAdaptation:
    """Stan's warmup windowing schedule for step-size + diagonal mass adaptation."""

    def __init__(self, n_warmup, dim):
        self.dim = dim
        if n_warmup >= 150:
            init_buffer = max(int(0.15 * n_warmup), 1)
            term_buffer = max(int(0.10 * n_warmup), 1)
            mid = n_warmup - init_buffer - term_buffer
            windows = []
            w = 25
            start = init_buffer
            while start + w < init_buffer + mid and w > 0:
                windows.append((start, start + w))
                start += w
                w *= 2
            windows.append((start, n_warmup - term_buffer))
            self.windows = windows
        else:
            self.windows = []
        self.welford = _Welford(dim)

    def is_window_end(self, i):
        for (a, b) in self.windows:
            if i + 1 == b:
                return True
        return False

    def in_window(self, i):
        for (a, b) in self.windows:
            if a <= i < b:
                return True
        return False

    def update(self, theta):
        if self.welford.n >= 0:
            self.welford.update(theta)

    def new_window_var(self, n_w):
        var = self.welford.var()
        return (n_w / (n_w + 5.0)) * var + 1e-3 * (5.0 / (n_w + 5.0))


class HamiltonianMonteCarlo(MCMCSampler):
    """Static-trajectory Hamiltonian Monte Carlo with leapfrog integration.

    ``mass_matrix``: ``None`` (identity, diagonal), a ``(dim,)`` diagonal vector, or a
    ``(dim, dim)`` full matrix. During warmup with ``adapt_step_size``/``adapt_mass``,
    the step size is tuned by dual averaging toward ``target_accept`` and (for diagonal
    mass) the mass matrix is re-estimated from the running variance at each warmup
    window's end (Stan's scheme); both freeze after warmup.
    """

    def __init__(self, log_prob, grad_log_prob=None, n_samples=1000, n_warmup=None,
                 n_chains=1, thin=1, step_size=0.1, n_leapfrog=10, mass_matrix=None,
                 adapt_step_size=True, adapt_mass=True, target_accept=0.65, jitter=0.0,
                 max_energy_change=1000.0):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.step_size = step_size
        self.n_leapfrog = int(n_leapfrog)
        self.mass_matrix = mass_matrix
        self.adapt_step_size = bool(adapt_step_size)
        self.adapt_mass = bool(adapt_mass) and (mass_matrix is None or np.ndim(mass_matrix) <= 1)
        self.target_accept = float(target_accept)
        self.jitter = float(jitter)
        self.max_energy_change = float(max_energy_change)

    def _init_state(self, theta, rng):
        logp = self.target(theta)
        grad = self.target.grad(theta)
        return {"theta": theta, "logp": logp, "grad": grad}

    def _start_chain(self, dim, rng):
        self._diag = self.mass_matrix is None or np.ndim(self.mass_matrix) <= 1
        if self.mass_matrix is None:
            self._inv_mass = np.ones(dim)
        elif np.ndim(self.mass_matrix) == 1:
            self._inv_mass = 1.0 / np.asarray(self.mass_matrix, dtype=float)
        else:
            M = np.asarray(self.mass_matrix, dtype=float)
            self._inv_mass = np.linalg.inv(M)
        self._eps = self.step_size if self.step_size is not None else 0.1
        self._dual = None
        self._adapt_win = _WindowedAdaptation(self.n_warmup, dim) if self.adapt_mass else None
        self._divergences = 0
        self._warmup_i = 0
        self._dim = dim
        if not hasattr(self, "_divergences_per_chain"):
            self._divergences_per_chain = []
        self._divergences_per_chain.append(0)

    def _find_init_step(self, theta, grad, rng):
        return _find_reasonable_step_size(self.target, theta, grad, self._inv_mass, rng, self._diag)

    def _step(self, state, rng, adapt):
        theta, logp, grad = state["theta"], state["logp"], state["grad"]
        if adapt and self.adapt_step_size and self._dual is None:
            eps0 = self._find_init_step(theta, grad, rng)
            self._dual = _DualAveraging(eps0, self.target_accept)
            self._eps = eps0
        p0 = _sample_momentum(rng, self._inv_mass, self._diag)
        H0 = -logp + _kinetic(p0, self._inv_mass)
        eps = self._eps * (1.0 + self.jitter * rng.uniform(-1, 1)) if self.jitter > 0 else self._eps
        theta1, p1, grad1, logp1 = _leapfrog(self.target, theta, p0, grad, eps, self._inv_mass, self.n_leapfrog, self._diag)
        H1 = -logp1 + _kinetic(p1, self._inv_mass)
        delta = H0 - H1
        divergent = (not np.isfinite(delta)) or (-delta > self.max_energy_change)
        accept_stat = 0.0 if divergent else min(1.0, float(np.exp(min(delta, 0.0))))
        if divergent:
            self._divergences += 1
            self._divergences_per_chain[-1] += 1
        accepted = (not divergent) and (np.log(rng.uniform()) < delta)
        new_state = {"theta": theta1, "logp": logp1, "grad": grad1} if accepted else state
        if adapt:
            if self.adapt_step_size:
                self._eps = self._dual.update(accept_stat)
            if self._adapt_win is not None:
                self._adapt_win.update(new_state["theta"])
                if self._adapt_win.is_window_end(self._warmup_i):
                    for (a, b) in self._adapt_win.windows:
                        if self._warmup_i + 1 == b:
                            n_w = b - a
                            self._inv_mass = self._adapt_win.new_window_var(n_w)
                            self._adapt_win.welford.reset()
                            if self.adapt_step_size:
                                self._eps = self._find_init_step(new_state["theta"], new_state["grad"], rng)
                                self._dual = _DualAveraging(self._eps, self.target_accept)
                            break
            self._warmup_i += 1
        return new_state, accept_stat

    def _end_warmup(self):
        if self.adapt_step_size and self._dual is not None:
            self._eps = self._dual.final()
        self.step_size_ = self._eps
        self.mass_matrix_ = (1.0 / self._inv_mass) if self._diag else np.linalg.inv(self._inv_mass)
        # divergences_ counts only the kept post-warmup transitions (transient warmup
        # divergences while step size/mass are still converging are not diagnostic).
        self._divergences = 0
        self._divergences_per_chain[-1] = 0

    def sample(self, theta_init, random_state=None):
        self._divergences_per_chain = []
        result = super().sample(theta_init, random_state)
        self.divergences_ = int(sum(self._divergences_per_chain))
        return result


class NoUTurnSampler(HamiltonianMonteCarlo):
    """No-U-Turn Sampler (Hoffman & Gelman 2014): multinomial trajectory sampling with
    the classic position-momentum stopping criterion (Algorithm 3), checked on each
    subtree's outer boundary. Recursively doubles the trajectory until a U-turn is
    detected (or ``max_treedepth`` is reached), sampling the next state with biased
    progressive sampling across the whole trajectory.
    """

    def __init__(self, log_prob, grad_log_prob=None, n_samples=1000, n_warmup=None,
                 n_chains=1, thin=1, step_size=None, max_treedepth=10, mass_matrix=None,
                 adapt_step_size=True, adapt_mass=True, target_accept=0.8,
                 max_energy_change=1000.0):
        super().__init__(log_prob, grad_log_prob, n_samples, n_warmup, n_chains, thin,
                          step_size=step_size if step_size is not None else 1.0,
                          n_leapfrog=1, mass_matrix=mass_matrix,
                          adapt_step_size=adapt_step_size, adapt_mass=adapt_mass,
                          target_accept=target_accept, jitter=0.0,
                          max_energy_change=max_energy_change)
        self.max_treedepth = int(max_treedepth)
        self._user_step_size = step_size

    def _start_chain(self, dim, rng):
        super()._start_chain(dim, rng)
        if self._user_step_size is not None:
            self._eps = self._user_step_size
        self._tree_depths = []

    def _uturn(self, theta_diff, p_minus, p_plus):
        """Classic NUTS stopping criterion (Hoffman & Gelman 2014, Algorithm 3), checked
        on a subtree's outer boundary: has the trajectory started to double back in
        velocity (``inv_mass @ p``) space relative to how far it has travelled in
        position?"""
        v_minus = self._inv_mass * p_minus if self._diag else self._inv_mass @ p_minus
        v_plus = self._inv_mass * p_plus if self._diag else self._inv_mass @ p_plus
        return (theta_diff @ v_minus < 0) or (theta_diff @ v_plus < 0)

    def _leaf(self, theta, p, grad, logp, H0, direction, eps):
        theta1, p1, grad1, logp1 = _leapfrog(self.target, theta, p, grad, direction * eps,
                                             self._inv_mass, 1, self._diag)
        H1 = -logp1 + _kinetic(p1, self._inv_mass)
        divergent = (not np.isfinite(H1)) or (H1 - H0 > self.max_energy_change)
        log_w = -H1 if not divergent else -np.inf
        accept = 0.0 if divergent else min(1.0, float(np.exp(min(H0 - H1, 0.0))))
        node = {
            "theta_minus": theta1, "p_minus": p1, "grad_minus": grad1,
            "theta_plus": theta1, "p_plus": p1, "grad_plus": grad1,
            "theta_prop": theta1, "grad_prop": grad1, "logp_prop": logp1,
            "log_sum_weight": log_w, "n_leapfrog": 1,
            "sum_accept": accept, "divergent": divergent, "turning": False,
        }
        return node

    def _build_tree(self, theta, p, grad, logp, H0, direction, depth, eps, rng):
        if depth == 0:
            return self._leaf(theta, p, grad, logp, H0, direction, eps)
        left = self._build_tree(theta, p, grad, logp, H0, direction, depth - 1, eps, rng)
        if left["divergent"] or left["turning"]:
            return left
        if direction == -1:
            other = self._build_tree(left["theta_minus"], left["p_minus"], left["grad_minus"],
                                     logp, H0, direction, depth - 1, eps, rng)
        else:
            other = self._build_tree(left["theta_plus"], left["p_plus"], left["grad_plus"],
                                     logp, H0, direction, depth - 1, eps, rng)
        if other["divergent"] or other["turning"]:
            combined = self._combine(left, other, direction, rng)
            combined["turning"] = True
            return combined
        return self._combine(left, other, direction, rng)

    def _combine(self, left, right, direction, rng):
        if direction == -1:
            outer_minus, outer_plus = right, left
        else:
            outer_minus, outer_plus = left, right
        log_sum = np.logaddexp(left["log_sum_weight"], right["log_sum_weight"])
        divergent = left["divergent"] or right["divergent"]
        if not divergent and np.isfinite(log_sum):
            # biased progressive sampling: P(pick right) = exp(right.log_w - log_sum)
            p_right = float(np.exp(min(right["log_sum_weight"] - log_sum, 0.0)))
            if rng.uniform() < p_right:
                theta_prop, grad_prop, logp_prop = right["theta_prop"], right["grad_prop"], right["logp_prop"]
            else:
                theta_prop, grad_prop, logp_prop = left["theta_prop"], left["grad_prop"], left["logp_prop"]
        else:
            theta_prop = left["theta_prop"]
            grad_prop = left["grad_prop"]
            logp_prop = left["logp_prop"]
        theta_diff = outer_plus["theta_plus"] - outer_minus["theta_minus"]
        turning = divergent or self._uturn(theta_diff, outer_minus["p_minus"], outer_plus["p_plus"])
        return {
            "theta_minus": outer_minus["theta_minus"], "p_minus": outer_minus["p_minus"],
            "grad_minus": outer_minus["grad_minus"],
            "theta_plus": outer_plus["theta_plus"], "p_plus": outer_plus["p_plus"],
            "grad_plus": outer_plus["grad_plus"],
            "theta_prop": theta_prop, "grad_prop": grad_prop, "logp_prop": logp_prop,
            "log_sum_weight": log_sum,
            "n_leapfrog": left["n_leapfrog"] + right["n_leapfrog"],
            "sum_accept": left["sum_accept"] + right["sum_accept"],
            "divergent": divergent, "turning": turning,
        }

    def _step(self, state, rng, adapt):
        theta, logp, grad = state["theta"], state["logp"], state["grad"]
        if adapt and self.adapt_step_size and self._dual is None:
            eps0 = self._user_step_size if self._user_step_size is not None else self._find_init_step(theta, grad, rng)
            self._dual = _DualAveraging(eps0, self.target_accept)
            self._eps = eps0
        p0 = _sample_momentum(rng, self._inv_mass, self._diag)
        H0 = -logp + _kinetic(p0, self._inv_mass)
        tree = {
            "theta_minus": theta, "p_minus": p0, "grad_minus": grad,
            "theta_plus": theta, "p_plus": p0, "grad_plus": grad,
            "theta_prop": theta, "grad_prop": grad, "logp_prop": logp,
            "log_sum_weight": -H0, "n_leapfrog": 0,
            "sum_accept": 0.0, "divergent": False, "turning": False,
        }
        depth = 0
        while depth < self.max_treedepth:
            direction = 1 if rng.integers(2) == 1 else -1
            if direction == -1:
                subtree = self._build_tree(tree["theta_minus"], tree["p_minus"], tree["grad_minus"],
                                           logp, H0, direction, depth, self._eps, rng)
            else:
                subtree = self._build_tree(tree["theta_plus"], tree["p_plus"], tree["grad_plus"],
                                           logp, H0, direction, depth, self._eps, rng)
            if subtree["divergent"] or subtree["turning"]:
                tree["n_leapfrog"] += subtree["n_leapfrog"]
                tree["sum_accept"] += subtree["sum_accept"]
                if subtree["divergent"]:
                    self._divergences += 1
                    self._divergences_per_chain[-1] += 1
                break
            merged = self._combine(tree, subtree, direction, rng) if direction == 1 else self._combine(subtree, tree, -1, rng)
            tree = merged
            depth += 1
            if tree["turning"]:
                break
        self._tree_depths.append(depth)
        accept_stat = tree["sum_accept"] / max(tree["n_leapfrog"], 1)
        new_state = {"theta": tree["theta_prop"], "logp": tree["logp_prop"], "grad": tree["grad_prop"]}
        if adapt:
            if self.adapt_step_size:
                self._eps = self._dual.update(accept_stat)
            if self._adapt_win is not None:
                self._adapt_win.update(new_state["theta"])
                if self._adapt_win.is_window_end(self._warmup_i):
                    for (a, b) in self._adapt_win.windows:
                        if self._warmup_i + 1 == b:
                            n_w = b - a
                            self._inv_mass = self._adapt_win.new_window_var(n_w)
                            self._adapt_win.welford.reset()
                            if self.adapt_step_size:
                                self._eps = self._find_init_step(new_state["theta"], new_state["grad"], rng)
                                self._dual = _DualAveraging(self._eps, self.target_accept)
                            break
            self._warmup_i += 1
        return new_state, accept_stat

    def sample(self, theta_init, random_state=None):
        result = super().sample(theta_init, random_state)
        n = self.n_samples * self.thin
        depths = np.array(self._tree_depths[-self.n_samples * self.thin:]) if self._tree_depths else np.array([0])
        # keep only the post-warmup depths, thinned like the chain itself
        kept = depths[self.thin - 1::self.thin][:self.n_samples] if self.thin > 1 else depths[:self.n_samples]
        if len(kept) < self.n_samples:
            kept = np.pad(kept, (0, self.n_samples - len(kept)), constant_values=(kept[-1] if len(kept) else 0))
        self.tree_depths_ = kept.reshape(1, -1) if self.n_chains == 1 else kept.reshape(self.n_chains, -1)
        self.mean_tree_depth_ = float(np.mean(self.tree_depths_))
        return result

    def _end_warmup(self):
        super()._end_warmup()
        self._tree_depths = []


class MALA(MCMCSampler):
    """Metropolis-adjusted Langevin algorithm: a gradient-drift Gaussian proposal
    corrected by a Metropolis accept/reject step, with dual-averaging step-size
    adaptation toward ``target_accept`` (Roberts & Rosenthal's optimal ~0.574).
    """

    def __init__(self, log_prob, grad_log_prob=None, n_samples=1000, n_warmup=None,
                 n_chains=1, thin=1, step_size=0.1, preconditioner=None,
                 adapt_step_size=True, target_accept=0.574):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.step_size = float(step_size)
        self.preconditioner = preconditioner
        self.adapt_step_size = bool(adapt_step_size)
        self.target_accept = float(target_accept)

    def _init_state(self, theta, rng):
        logp = self.target(theta)
        grad = self.target.grad(theta)
        return {"theta": theta, "logp": logp, "grad": grad}

    def _start_chain(self, dim, rng):
        self._P = np.ones(dim) if self.preconditioner is None else np.asarray(self.preconditioner, dtype=float)
        self._eps = self.step_size
        self._dual = _DualAveraging(self._eps, self.target_accept)

    @staticmethod
    def _mean(theta, grad, eps, P):
        return theta + 0.5 * eps ** 2 * P * grad

    def _log_q(self, to, mean_, eps, P):
        diff = to - mean_
        return float(-0.5 * np.sum(diff ** 2 / (eps ** 2 * P)))

    def _step(self, state, rng, adapt):
        theta, logp, grad = state["theta"], state["logp"], state["grad"]
        mean_fwd = self._mean(theta, grad, self._eps, self._P)
        new = mean_fwd + self._eps * np.sqrt(self._P) * rng.standard_normal(len(theta))
        logp_new = self.target(new)
        grad_new = self.target.grad(new)
        mean_bwd = self._mean(new, grad_new, self._eps, self._P)
        log_q_fwd = self._log_q(new, mean_fwd, self._eps, self._P)
        log_q_bwd = self._log_q(theta, mean_bwd, self._eps, self._P)
        log_alpha = (logp_new - logp) + (log_q_bwd - log_q_fwd)
        accept_stat = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
        accepted = _log_accept(np.log(rng.uniform()), log_alpha)
        new_state = {"theta": new, "logp": logp_new, "grad": grad_new} if accepted else state
        if adapt and self.adapt_step_size:
            self._eps = self._dual.update(accept_stat)
        return new_state, accept_stat

    def _end_warmup(self):
        if self.adapt_step_size:
            self._eps = self._dual.final()
        self.step_size_ = self._eps


def _softabs_metric(H, alpha):
    lam, Q = np.linalg.eigh(H)
    lam_soft = np.where(np.abs(lam) < 1e-12, 1.0 / alpha, lam / np.tanh(alpha * lam))
    return Q @ np.diag(lam_soft) @ Q.T


class MMALA(MCMCSampler):
    """Simplified manifold MALA (Girolami & Calderhead 2011) with a position-dependent
    metric and no Christoffel-symbol correction term.

    ``metric(theta) -> (dim, dim)`` SPD; defaults to the SoftAbs-regularised negative
    Hessian of ``log_prob`` (finite-differenced if no analytic Hessian is supplied).
    """

    def __init__(self, log_prob, grad_log_prob=None, metric=None, n_samples=1000,
                 n_warmup=None, n_chains=1, thin=1, step_size=0.5, adapt_step_size=True,
                 target_accept=0.574, softabs_alpha=1e6):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.metric_fn = metric
        self.step_size = float(step_size)
        self.adapt_step_size = bool(adapt_step_size)
        self.target_accept = float(target_accept)
        self.softabs_alpha = float(softabs_alpha)

    def _metric(self, theta):
        if self.metric_fn is not None:
            return np.asarray(self.metric_fn(theta), dtype=float)
        H = self.target.hessian(theta)
        return _softabs_metric(-H, self.softabs_alpha)

    def _init_state(self, theta, rng):
        logp = self.target(theta)
        grad = self.target.grad(theta)
        G = self._metric(theta)
        return {"theta": theta, "logp": logp, "grad": grad, "G": G}

    def _start_chain(self, dim, rng):
        self._eps = self.step_size
        self._dual = _DualAveraging(self._eps, self.target_accept)

    def _mean(self, theta, grad, G_inv, eps):
        return theta + 0.5 * eps ** 2 * G_inv @ grad

    def _step(self, state, rng, adapt):
        theta, logp, grad, G = state["theta"], state["logp"], state["grad"], state["G"]
        G_inv = np.linalg.inv(G)
        L = _cholesky_psd(G_inv)
        mean_fwd = self._mean(theta, grad, G_inv, self._eps)
        new = mean_fwd + self._eps * (L @ rng.standard_normal(len(theta)))
        logp_new = self.target(new)
        grad_new = self.target.grad(new)
        G_new = self._metric(new)
        G_new_inv = np.linalg.inv(G_new)
        mean_bwd = self._mean(new, grad_new, G_new_inv, self._eps)
        sign_old, logdet_old = np.linalg.slogdet(G)
        sign_new, logdet_new = np.linalg.slogdet(G_new)
        diff_fwd = new - mean_fwd
        diff_bwd = theta - mean_bwd
        log_q_fwd = -0.5 * diff_fwd @ G @ diff_fwd / self._eps ** 2 + 0.5 * logdet_old
        log_q_bwd = -0.5 * diff_bwd @ G_new @ diff_bwd / self._eps ** 2 + 0.5 * logdet_new
        log_alpha = (logp_new - logp) + (log_q_bwd - log_q_fwd)
        accept_stat = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
        accepted = _log_accept(np.log(rng.uniform()), log_alpha)
        new_state = {"theta": new, "logp": logp_new, "grad": grad_new, "G": G_new} if accepted else state
        if adapt and self.adapt_step_size:
            self._eps = self._dual.update(accept_stat)
        return new_state, accept_stat

    def _end_warmup(self):
        if self.adapt_step_size:
            self._eps = self._dual.final()
        self.step_size_ = self._eps


class RiemannianHMC(MCMCSampler):
    """Riemannian manifold HMC (Girolami & Calderhead 2011) with the generalized
    (implicit midpoint) leapfrog integrator solved by fixed-point iteration.

    ``metric(theta) -> (dim, dim)`` SPD (default: SoftAbs of the negative Hessian, as in
    :class:`MMALA`); ``metric_grad(theta) -> (dim, dim, dim)`` with axis 0 indexing
    ``dG/dtheta_i`` (default: central finite differences of ``metric``). Cost is
    ``O(dim**3)`` per fixed-point iteration -- intended for modest dimension.
    """

    def __init__(self, log_prob, grad_log_prob=None, metric=None, metric_grad=None,
                 n_samples=1000, n_warmup=None, n_chains=1, thin=1, step_size=0.1,
                 n_leapfrog=5, n_fixed_point=5, adapt_step_size=True, target_accept=0.8,
                 softabs_alpha=1e6, fd_eps=1e-5):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.metric_fn = metric
        self.metric_grad_fn = metric_grad
        self.step_size = float(step_size)
        self.n_leapfrog = int(n_leapfrog)
        self.n_fixed_point = int(n_fixed_point)
        self.adapt_step_size = bool(adapt_step_size)
        self.target_accept = float(target_accept)
        self.softabs_alpha = float(softabs_alpha)
        self.fd_eps = float(fd_eps)

    def _metric(self, theta):
        if self.metric_fn is not None:
            return np.asarray(self.metric_fn(theta), dtype=float)
        H = self.target.hessian(theta)
        return _softabs_metric(-H, self.softabs_alpha)

    def _metric_grad(self, theta):
        dim = len(theta)
        if self.metric_grad_fn is not None:
            return np.asarray(self.metric_grad_fn(theta), dtype=float)
        dG = np.empty((dim, dim, dim))
        eps = self.fd_eps
        for i in range(dim):
            step = eps * max(1.0, abs(theta[i]))
            tp = theta.copy(); tp[i] += step
            tm = theta.copy(); tm[i] -= step
            dG[i] = (self._metric(tp) - self._metric(tm)) / (2 * step)
        return dG

    def _H(self, theta, p, G):
        sign, logdet = np.linalg.slogdet(G)
        G_inv = np.linalg.inv(G)
        return -self.target(theta) + 0.5 * logdet + 0.5 * float(p @ G_inv @ p)

    def _dH_dtheta(self, theta, p, G, G_inv, dG):
        grad = -self.target.grad(theta)
        dim = len(theta)
        out = grad.copy()
        for i in range(dim):
            out[i] += 0.5 * np.trace(G_inv @ dG[i])
            out[i] -= 0.5 * float(p @ G_inv @ dG[i] @ G_inv @ p)
        return out

    def _generalized_leapfrog(self, theta, p, eps, rng):
        theta = theta.copy()
        p = p.copy()
        G = self._metric(theta)
        for _ in range(self.n_leapfrog):
            dG = self._metric_grad(theta)
            G_inv = np.linalg.inv(G)
            p_half = p.copy()
            for _ in range(self.n_fixed_point):
                dHdtheta = self._dH_dtheta(theta, p_half, G, G_inv, dG)
                p_half = p - 0.5 * eps * dHdtheta
            theta_new = theta.copy()
            G_new = G
            G_new_inv = G_inv
            for _ in range(self.n_fixed_point):
                v_theta = G_inv @ p_half
                v_new = G_new_inv @ p_half
                theta_new = theta + 0.5 * eps * (v_theta + v_new)
                G_new = self._metric(theta_new)
                G_new_inv = np.linalg.inv(G_new)
            dG_new = self._metric_grad(theta_new)
            dHdtheta_new = self._dH_dtheta(theta_new, p_half, G_new, G_new_inv, dG_new)
            p_new = p_half - 0.5 * eps * dHdtheta_new
            theta, p, G = theta_new, p_new, G_new
        return theta, p, G

    def _init_state(self, theta, rng):
        logp = self.target(theta)
        G = self._metric(theta)
        return {"theta": theta, "logp": logp, "G": G}

    def _start_chain(self, dim, rng):
        self._eps = self.step_size
        self._dual = _DualAveraging(self._eps, self.target_accept)
        self._dim = dim

    def _step(self, state, rng, adapt):
        theta, logp, G = state["theta"], state["logp"], state["G"]
        L = _cholesky_psd(G)
        p0 = L @ rng.standard_normal(self._dim)
        H0 = self._H(theta, p0, G)
        try:
            theta1, p1, G1 = self._generalized_leapfrog(theta, p0, self._eps, rng)
            H1 = self._H(theta1, p1, G1)
        except np.linalg.LinAlgError:
            H1 = np.inf
            theta1, G1 = theta, G
        log_alpha = H0 - H1
        accept_stat = min(1.0, float(np.exp(min(log_alpha, 0.0)))) if np.isfinite(log_alpha) else 0.0
        accepted = _log_accept(np.log(rng.uniform()), log_alpha)
        new_state = {"theta": theta1, "logp": self.target(theta1), "G": G1} if accepted else state
        if adapt and self.adapt_step_size:
            self._eps = self._dual.update(accept_stat)
        return new_state, accept_stat

    def _end_warmup(self):
        if self.adapt_step_size:
            self._eps = self._dual.final()
        self.step_size_ = self._eps


class NeutraHMC(MCMCSampler):
    """Neural transport HMC (Hoffman, Sountsov, Dillon, Langmore, Tran & Vasudevan 2019):
    fit a :class:`~stochpylib.advanced_mcmc.variational.NormalizingFlows` approximation of
    the target, then run NUTS (default) or HMC in the flow's standard-normal base space
    and push the draws back through the flow -- often much better conditioned than
    sampling the original (correlated / curved) target directly.
    """

    def __init__(self, log_prob, grad_log_prob=None, n_samples=1000, n_warmup=None,
                 n_chains=1, thin=1, flow=None, flow_layers=8, flow_iter=1500,
                 flow_lr=0.01, inner="nuts", inner_kwargs=None):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin, grad_log_prob)
        self.flow = flow
        self.flow_layers = int(flow_layers)
        self.flow_iter = int(flow_iter)
        self.flow_lr = float(flow_lr)
        self.inner = inner
        self.inner_kwargs = dict(inner_kwargs) if inner_kwargs else {}

    def sample(self, theta_init, random_state=None):
        from stochpylib.advanced_mcmc._base import LogDensity
        from stochpylib.advanced_mcmc._common import _rng as rng_fn
        from stochpylib.advanced_mcmc.variational import NormalizingFlows

        rng = rng_fn(random_state)
        dim = np.atleast_1d(np.asarray(theta_init, dtype=float)).shape[-1]

        flow = self.flow
        if flow is None:
            flow = NormalizingFlows(self.target, dim, n_layers=self.flow_layers,
                                    n_iter=self.flow_iter, lr=self.flow_lr)
            flow.fit(random_state=rng)
        self.flow_ = flow

        def z_log_prob(z):
            theta, logdet = flow.forward(z)
            return self.target(theta) + logdet

        def z_grad(z):
            theta, logdet = flow.forward(z)
            grad_theta = self.target.grad(theta)
            return flow.pullback_grad(z, grad_theta)

        z_target = LogDensity(z_log_prob, z_grad)
        if self.inner == "nuts":
            inner = NoUTurnSampler(z_target.log_prob, z_target.grad_log_prob,
                                   n_samples=self.n_samples, n_warmup=self.n_warmup,
                                   n_chains=self.n_chains, thin=self.thin, **self.inner_kwargs)
        else:
            inner = HamiltonianMonteCarlo(z_target.log_prob, z_target.grad_log_prob,
                                          n_samples=self.n_samples, n_warmup=self.n_warmup,
                                          n_chains=self.n_chains, thin=self.thin, **self.inner_kwargs)
        z_init = 0.1 * rng.standard_normal((self.n_chains, dim))
        inner.sample(z_init, random_state=rng)
        self.inner_ = inner
        self.z_chains_ = inner.chains_
        chains = np.empty_like(inner.chains_)
        logps = np.empty(inner.chains_.shape[:2])
        for c in range(inner.chains_.shape[0]):
            for i in range(inner.chains_.shape[1]):
                theta, _ = flow.forward(inner.chains_[c, i])
                chains[c, i] = theta
                logps[c, i] = self.target(theta)
        self.chains_ = chains
        self.log_probs_ = logps
        self.acceptance_rates_ = inner.acceptance_rates_
        self.acceptance_rate_ = inner.acceptance_rate_
        self.dim_ = dim
        return self
