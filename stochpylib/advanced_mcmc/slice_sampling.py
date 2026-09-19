"""Slice sampling: Neal's (2003) univariate stepping-out/doubling schemes, Murray, Adams &
MacKay's (2010) elliptical slice sampler for Gaussian-prior models, and the Gibbsian polar
slice sampler of Schar, Habeck & Rudolf (2023).
"""

import numpy as np

from stochpylib.advanced_mcmc._base import MCMCSampler
from stochpylib.advanced_mcmc._common import _cholesky_psd

__all__ = ["Doubling", "EllipticalSliceSampling", "Polar_Slice", "SliceSampling", "Stepping"]


class Stepping:
    """Neal (2003) Figure 3: stepping-out interval search for slice sampling."""

    def __init__(self, width=1.0, max_steps=100):
        self.width = float(width)
        self.max_steps = int(max_steps)

    def find_interval(self, f, x0, log_y, rng):
        u = rng.uniform()
        L = x0 - self.width * u
        R = L + self.width
        j = int(np.floor(self.max_steps * rng.uniform()))
        k = self.max_steps - 1 - j
        while j > 0 and f(L) > log_y:
            L -= self.width
            j -= 1
        while k > 0 and f(R) > log_y:
            R += self.width
            k -= 1
        return L, R

    def accept(self, f, x0, x1, log_y, L, R):
        return True


class Doubling:
    """Neal (2003) Figure 4/6: doubling interval search with the matching acceptance test."""

    def __init__(self, width=1.0, max_doublings=20):
        self.width = float(width)
        self.max_doublings = int(max_doublings)

    def find_interval(self, f, x0, log_y, rng):
        u = rng.uniform()
        L = x0 - self.width * u
        R = L + self.width
        k = self.max_doublings
        while k > 0 and (f(L) > log_y or f(R) > log_y):
            if rng.uniform() < 0.5:
                L -= (R - L)
            else:
                R += (R - L)
            k -= 1
        return L, R

    def accept(self, f, x0, x1, log_y, L, R):
        Lh, Rh = L, R
        D = False
        while Rh - Lh > 1.1 * self.width:
            M = (Lh + Rh) / 2.0
            if (x0 < M <= x1) or (x1 < M <= x0):
                D = True
            if x1 < M:
                Rh = M
            else:
                Lh = M
            if D and f(Lh) <= log_y and f(Rh) <= log_y:
                return False
        return True


class SliceSampling(MCMCSampler):
    """Coordinate-wise univariate slice sampling (Neal 2003) with shrinkage.

    ``interval``: ``"stepping"`` (default), ``"doubling"``, or a :class:`Stepping`/
    :class:`Doubling` instance. ``width`` is a scalar or a ``(dim,)`` vector of per-
    coordinate slice widths.
    """

    def __init__(self, log_prob, n_samples=1000, n_warmup=None, n_chains=1, thin=1,
                 width=1.0, interval="stepping", max_steps=100):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin)
        if isinstance(interval, str):
            if interval == "stepping":
                self._finder_template = Stepping(1.0, max_steps)
            elif interval == "doubling":
                self._finder_template = Doubling(1.0, max_steps)
            else:
                raise ValueError("interval must be 'stepping', 'doubling', or a finder instance")
        else:
            self._finder_template = interval
        self.width = width

    def _start_chain(self, dim, rng):
        w = np.asarray(self.width, dtype=float)
        self._width_vec = np.broadcast_to(w, (dim,)).copy() if w.ndim == 0 else w

    def _step(self, state, rng, adapt):
        theta = state["theta"].copy()
        logp = state["logp"]
        for i in range(len(theta)):
            width_i = self._width_vec[i]
            finder = _clone_finder(self._finder_template, width_i)

            def f(v, i=i, base=theta):
                t = base.copy()
                t[i] = v
                return self.target(t)

            log_y = logp - rng.exponential()
            L, R = finder.find_interval(f, theta[i], log_y, rng)
            for _ in range(1000):
                x1 = L + rng.uniform() * (R - L)
                lp1 = f(x1)
                if lp1 > log_y and finder.accept(f, theta[i], x1, log_y, L, R):
                    theta[i] = x1
                    logp = lp1
                    break
                if x1 < theta[i]:
                    L = x1
                else:
                    R = x1
            else:
                raise RuntimeError("slice sampling shrinkage failed to find an acceptable point")
        return {"theta": theta, "logp": logp}, 1.0


def _clone_finder(template, width):
    if isinstance(template, Stepping):
        return Stepping(width, template.max_steps)
    return Doubling(width, template.max_doublings)


class EllipticalSliceSampling(MCMCSampler):
    """Murray, Adams & MacKay (2010): exact slice sampling for a Gaussian prior with an
    arbitrary likelihood. ``log_prob`` here is the log-*likelihood* only; the Gaussian
    prior is given by ``prior_cov`` (or ``prior_chol`` directly) and ``prior_mean``.
    """

    def __init__(self, log_likelihood, prior_cov=None, prior_chol=None, prior_mean=None,
                 n_samples=1000, n_warmup=None, n_chains=1, thin=1):
        super().__init__(log_likelihood, n_samples, n_warmup, n_chains, thin)
        if prior_chol is None and prior_cov is None:
            raise ValueError("either prior_cov or prior_chol must be given")
        self.prior_cov = prior_cov
        self.prior_chol = prior_chol
        self.prior_mean = prior_mean

    def _start_chain(self, dim, rng):
        self._L = self.prior_chol if self.prior_chol is not None else _cholesky_psd(np.asarray(self.prior_cov, dtype=float))
        self._mean = np.zeros(dim) if self.prior_mean is None else np.asarray(self.prior_mean, dtype=float)

    def _step(self, state, rng, adapt):
        theta, loglik = state["theta"], state["logp"]
        nu = self._mean + self._L @ rng.standard_normal(len(theta))
        log_y = loglik + np.log(rng.uniform())
        angle = rng.uniform(0.0, 2.0 * np.pi)
        angle_min, angle_max = angle - 2.0 * np.pi, angle
        centered = theta - self._mean
        for _ in range(1000):
            new = self._mean + centered * np.cos(angle) + (nu - self._mean) * np.sin(angle)
            loglik_new = self.target(new)
            if loglik_new > log_y:
                return {"theta": new, "logp": loglik_new}, 1.0
            if angle < 0:
                angle_min = angle
            else:
                angle_max = angle
            angle = angle_min + rng.uniform() * (angle_max - angle_min)
        raise RuntimeError("elliptical slice sampling shrinkage failed to find an acceptable point")


class Polar_Slice(MCMCSampler):
    """Gibbsian polar slice sampler (Schar, Habeck & Rudolf 2023) for targets on
    ``R^d`` (``d >= 2``) that vanish or are well-behaved at the origin. Samples the level
    for ``f1(x) = ||x||^(d-1) p(x)``, then updates direction (great-circle shrinkage) and
    radius (1-D stepping-out/shrinkage) in turn.
    """

    def __init__(self, log_prob, n_samples=1000, n_warmup=None, n_chains=1, thin=1,
                 width=1.0, max_steps=100):
        super().__init__(log_prob, n_samples, n_warmup, n_chains, thin)
        self.width = float(width)
        self.max_steps = int(max_steps)

    def _init_state(self, theta, rng):
        theta = np.asarray(theta, dtype=float)
        if len(theta) < 2:
            raise ValueError("Polar_Slice requires dimension >= 2 (use SliceSampling for dim=1)")
        r = np.linalg.norm(theta)
        if r < 1e-12:
            theta = theta + 1e-6
            r = np.linalg.norm(theta)
        logp = self.target(theta)
        return {"theta": theta, "logp": logp, "r": r}

    def _log_f1(self, x, d):
        r = np.linalg.norm(x)
        if r < 1e-300:
            return -np.inf
        return (d - 1) * np.log(r) + self.target(x)

    def _step(self, state, rng, adapt):
        theta = state["theta"]
        d = len(theta)
        r = np.linalg.norm(theta)
        direction = theta / r
        log_f1 = self._log_f1(theta, d)
        log_y = log_f1 - rng.exponential()

        # direction update: great-circle shrinkage against a random orthogonal direction
        v = rng.standard_normal(d)
        v = v - (v @ direction) * direction
        vnorm = np.linalg.norm(v)
        if vnorm < 1e-300:
            v = np.eye(d)[0] - (np.eye(d)[0] @ direction) * direction
            vnorm = np.linalg.norm(v)
        v = v / vnorm
        phi = rng.uniform(0.0, 2.0 * np.pi)
        phi_min, phi_max = phi - 2.0 * np.pi, phi
        new_dir = direction
        for _ in range(1000):
            cand_dir = direction * np.cos(phi) + v * np.sin(phi)
            cand_dir = cand_dir / np.linalg.norm(cand_dir)
            if self._log_f1(r * cand_dir, d) > log_y:
                new_dir = cand_dir
                break
            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi
            phi = phi_min + rng.uniform() * (phi_max - phi_min)
        else:
            raise RuntimeError("Polar_Slice direction shrinkage failed")

        # radius update: 1-D stepping-out + shrinkage on g(s) = (d-1) log s + logp(s * new_dir)
        def g(s):
            if s <= 0:
                return -np.inf
            return (d - 1) * np.log(s) + self.target(s * new_dir)

        log_y_r = g(r) - rng.exponential()
        finder = Stepping(self.width, self.max_steps)
        L, R = finder.find_interval(g, r, log_y_r, rng)
        L = max(L, 1e-300)
        new_r = r
        for _ in range(1000):
            s1 = L + rng.uniform() * (R - L)
            if g(s1) > log_y_r:
                new_r = s1
                break
            if s1 < r:
                L = s1
            else:
                R = s1
        else:
            raise RuntimeError("Polar_Slice radius shrinkage failed")

        new_theta = new_r * new_dir
        return {"theta": new_theta, "logp": self.target(new_theta), "r": new_r}, 1.0
