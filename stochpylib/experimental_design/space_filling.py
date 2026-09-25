"""Space-filling designs for computer experiments: Latin hypercubes (plain, centred,
maximin/correlation-screened), Morris-Mitchell maximin LHDs, minimax (fill-distance)
designs, uniform designs minimizing an L2 discrepancy, and orthogonal-array designs (with
Tang's OA-based Latin hypercube). All points live in the unit cube ``[0, 1]^d``
(``Design.space == "unit"``); :meth:`Design.to_natural` maps them onto real bounds.
"""

import math

import numpy as np

from stochpylib.experimental_design._base import DesignGenerator
from stochpylib.experimental_design._common import (
    _discrepancy,
    _gf_tables,
    _is_lhs,
    _is_prime_power,
    _max_abs_correlation,
    _min_distance,
    _pairwise,
    _sobol_reference,
)
from stochpylib.experimental_design._design import Design

__all__ = ["LatinHypercubeDesign", "OrthogonalArrayDesign", "UniformDesign", "MaximinLHD",
           "MinimaxDesign"]


def _lhs(n, dim, rng):
    from stochpylib.montecarlo import LatinHypercubeSampling

    return LatinHypercubeSampling(dim=dim, n=n, random_state=rng).generate()


def _centered(U):
    n = len(U)
    return (np.clip(np.floor(U * n), 0, n - 1) + 0.5) / n


def _unit_design(U, kind, bounds, props):
    return Design(U, space="unit", kind=kind, bounds=bounds, properties=props)


class LatinHypercubeDesign(DesignGenerator):
    """Latin hypercube design: each of the ``n`` equal strata of every factor is hit once.

    Draws through :class:`stochpylib.montecarlo.LatinHypercubeSampling`. ``centered=True``
    moves every point to its stratum midpoint. ``criterion="maximin"`` (largest minimum
    distance) or ``"correlation"`` (smallest maximum absolute column correlation) keeps
    the best of ``n_iter`` draws.
    """

    kind = "Latin hypercube"

    def __init__(self, n, dim, centered=False, criterion=None, n_iter=100, bounds=None,
                 random_state=None):
        self.n, self.dim = int(n), int(dim)
        self.centered = bool(centered)
        if criterion not in (None, "maximin", "correlation"):
            raise ValueError("criterion must be None, 'maximin' or 'correlation'")
        self.criterion = criterion
        self.n_iter = int(n_iter)
        self.bounds = bounds
        self.random_state = random_state

    def _draw(self, rng):
        U = _lhs(self.n, self.dim, rng)
        return _centered(U) if self.centered else U

    def _generate(self, rng):
        if self.criterion is None:
            U = self._draw(rng)
        else:
            best, best_val = None, -np.inf
            for _ in range(max(1, self.n_iter)):
                U = self._draw(rng)
                val = (_min_distance(U) if self.criterion == "maximin"
                       else -_max_abs_correlation(U))
                if val > best_val:
                    best, best_val = U, val
            U = best
        return _unit_design(U, self.kind, self.bounds,
                            {"min_distance": _min_distance(U),
                             "max_abs_correlation": _max_abs_correlation(U)})


class MaximinLHD(DesignGenerator):
    """Maximin Latin hypercube by Morris & Mitchell (1995) simulated annealing.

    Minimizes ``phi_p = (sum_{i<j} d_ij^-p)^(1/p)``, which for large ``p`` is equivalent
    to maximizing the minimum distance while also breaking ties on how many pairs sit at
    it. A move swaps two entries of one column, so every iterate stays a Latin hypercube.
    """

    kind = "maximin Latin hypercube"

    def __init__(self, n, dim, p=15, n_iter=2000, t0=None, cooling=0.95, bounds=None,
                 random_state=None):
        self.n, self.dim = int(n), int(dim)
        if self.n < 2:
            raise ValueError("MaximinLHD needs n >= 2")
        self.p = float(p)
        self.n_iter = int(n_iter)
        self.t0 = t0
        self.cooling = float(cooling)
        self.bounds = bounds
        self.random_state = random_state

    def _phi(self, D2):
        d = np.sqrt(np.maximum(D2[np.triu_indices(self.n, 1)], 1e-24))
        # scale by the minimum first so d^-p cannot overflow for large p
        m = d.min()
        return float(np.sum((d / m) ** -self.p) ** (1.0 / self.p) / m)

    def _generate(self, rng):
        U = _centered(_lhs(self.n, self.dim, rng))
        D2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
        phi = self._phi(D2)
        phi0 = phi
        best, best_phi = U.copy(), phi
        # a starting temperature on the scale of a typical move's change in phi_p
        T = float(self.t0) if self.t0 is not None else 0.1 * phi
        steps_per_temp = max(10, self.n_iter // 50)
        for it in range(self.n_iter):
            j = int(rng.integers(self.dim))
            a, b = rng.choice(self.n, 2, replace=False)
            V = U.copy()
            V[a, j], V[b, j] = U[b, j], U[a, j]
            # a swap in one column only changes the distances of rows a and b
            E2 = D2.copy()
            for r in (a, b):
                row = ((V[r] - V) ** 2).sum(1)
                E2[r, :] = row
                E2[:, r] = row
            new = self._phi(E2)
            if new < phi or rng.random() < math.exp(-(new - phi) / max(T, 1e-300)):
                U, D2, phi = V, E2, new
                if phi < best_phi:
                    best, best_phi = U.copy(), phi
            if (it + 1) % steps_per_temp == 0:
                T *= self.cooling
        return _unit_design(best, self.kind, self.bounds,
                            {"phi_p": best_phi, "phi_p_initial": phi0, "p": self.p,
                             "min_distance": _min_distance(best)})


def _enclosing_center(P, n_steps=50):
    """Approximate minimum-enclosing-ball centre (Badoiu-Clarkson core-set iteration)."""
    c = P.mean(axis=0)
    for t in range(1, n_steps + 1):
        far = P[int(np.argmax(((P - c) ** 2).sum(1)))]
        c = c + (far - c) / (t + 1)
    return c


class MinimaxDesign(DesignGenerator):
    """Minimax design: minimizes the fill distance (the largest distance from any point of
    the cube to its nearest run).

    The cube is represented by ``n_ref`` Sobol reference points. The design starts from a
    k-means (Lloyd) clustering of the references and then repeatedly moves every run to
    the approximate minimum-enclosing-ball centre of its Voronoi cell -- the minimax
    analogue of Lloyd's centroid step -- keeping the best iterate. The reported
    ``fill_distance`` is measured on the reference set.
    """

    kind = "minimax"

    def __init__(self, n, dim, n_ref=4096, n_iter=50, bounds=None, random_state=None):
        self.n, self.dim = int(n), int(dim)
        self.n_ref = int(n_ref)
        self.n_iter = int(n_iter)
        self.bounds = bounds
        self.random_state = random_state

    def _assign(self, ref, X):
        lab = np.empty(len(ref), dtype=int)
        dist = np.empty(len(ref))
        for s in range(0, len(ref), 2048):
            D = _pairwise(ref[s:s + 2048], X)
            lab[s:s + 2048] = D.argmin(1)
            dist[s:s + 2048] = D.min(1)
        return lab, dist

    def _generate(self, rng):
        ref = _sobol_reference(self.dim, self.n_ref, rng)
        X = ref[rng.choice(len(ref), self.n, replace=False)].copy()
        for _ in range(20):  # Lloyd warm start
            lab, _ = self._assign(ref, X)
            for c in range(self.n):
                if np.any(lab == c):
                    X[c] = ref[lab == c].mean(0)
        lab, dist = self._assign(ref, X)
        best, best_fill, stale = X.copy(), float(dist.max()), 0
        for _ in range(self.n_iter):
            for c in range(self.n):
                cell = ref[lab == c]
                if len(cell):
                    X[c] = _enclosing_center(cell)
            X = np.clip(X, 0.0, 1.0)
            lab, dist = self._assign(ref, X)
            fill = float(dist.max())
            if fill < best_fill * (1 - 1e-6):
                best, best_fill, stale = X.copy(), fill, 0
            else:
                stale += 1
                if stale >= 5:
                    break
        return _unit_design(best, self.kind, self.bounds,
                            {"fill_distance": best_fill, "n_ref": self.n_ref,
                             "min_distance": _min_distance(best)})


class UniformDesign(DesignGenerator):
    """Uniform design (Fang & Wang): a U-type design minimizing an L2 discrepancy.

    ``method="glp"`` searches the good-lattice-point (Korobov) generators
    ``h = (1, a, a^2, ...) mod n`` and keeps the one with the smallest ``criterion``
    discrepancy (``"CD"`` centred, ``"WD"`` wrap-around, ``"MD"`` mixture, ``"L2-star"``);
    ``method="lhs-search"`` -- also the fallback when no admissible generator exists --
    runs threshold-accepting column swaps from a centred Latin hypercube.
    """

    kind = "uniform design"

    def __init__(self, n, dim, criterion="CD", method="glp", n_iter=2000, bounds=None,
                 random_state=None):
        self.n, self.dim = int(n), int(dim)
        if criterion not in ("CD", "WD", "MD", "L2-star"):
            raise ValueError("criterion must be 'CD', 'WD', 'MD' or 'L2-star'")
        if method not in ("glp", "lhs-search"):
            raise ValueError("method must be 'glp' or 'lhs-search'")
        self.criterion = criterion
        self.method = method
        self.n_iter = int(n_iter)
        self.bounds = bounds
        self.random_state = random_state

    @staticmethod
    def discrepancy(X, method="CD"):
        """L2-type discrepancy of points in the unit cube (squared for CD/WD/MD, matching
        ``scipy.stats.qmc.discrepancy``'s conventions)."""
        return _discrepancy(X, method)

    def _glp(self):
        n, d = self.n, self.dim
        i = np.arange(1, n + 1)[:, None]
        best, best_val, best_a = None, np.inf, None
        for a in range(2, n):
            h = [pow(a, j, n) for j in range(d)]
            if len(set(h)) < d or any(math.gcd(v, n) != 1 for v in h):
                continue
            U = ((i * np.array(h)[None, :]) % n + 0.5) / n
            val = _discrepancy(U, self.criterion)
            if val < best_val:
                best, best_val, best_a = U, val, a
        if d == 1 and best is None:
            best = ((np.arange(n) + 0.5) / n)[:, None]
            best_val, best_a = _discrepancy(best, self.criterion), 1
        return best, best_val, best_a

    def _search(self, rng):
        U = _centered(_lhs(self.n, self.dim, rng))
        val = _discrepancy(U, self.criterion)
        best, best_val = U.copy(), val
        threshold = 0.05 * val
        for it in range(self.n_iter):
            j = int(rng.integers(self.dim))
            a, b = rng.choice(self.n, 2, replace=False)
            V = U.copy()
            V[a, j], V[b, j] = U[b, j], U[a, j]
            new = _discrepancy(V, self.criterion)
            if new < val + threshold:
                U, val = V, new
                if val < best_val:
                    best, best_val = U.copy(), val
            threshold *= 0.998
        return best, best_val

    def _generate(self, rng):
        props = {"criterion": self.criterion}
        U = None
        if self.method == "glp":
            U, val, a = self._glp()
            if U is not None:
                props.update(method="glp", generator=a)
        if U is None:
            U, val = self._search(rng)
            props["method"] = "lhs-search"
        props["discrepancy"] = float(val)
        return _unit_design(U, self.kind, self.bounds, props)


class OrthogonalArrayDesign(DesignGenerator):
    """Orthogonal-array design OA(s^t, n_factors, s, t) by Bush's construction.

    Every set of ``strength`` columns contains each of the ``s^t`` level combinations
    equally often (index ``lambda = N / s^t``; here ``N = s^t`` so ``lambda = 1``). ``s``
    must be a prime power (the construction evaluates polynomials of degree < ``t`` over
    GF(s)); up to ``s + 1`` factors. Levels map to the unit cube as ``(level + 0.5)/s``.
    ``lhs=True`` returns Tang's (1993) OA-based Latin hypercube instead, which keeps the
    array's projections balanced while stratifying every margin into ``N`` cells.
    """

    kind = "orthogonal array"

    def __init__(self, levels, n_factors, strength=2, lhs=False, bounds=None,
                 random_state=None):
        self.levels = int(levels)
        self.n_factors = int(n_factors)
        self.strength = int(strength)
        self.lhs = bool(lhs)
        self.bounds = bounds
        self.random_state = random_state
        if _is_prime_power(self.levels) is None:
            raise ValueError("levels must be a prime power (2, 3, 4, 5, 7, 8, 9, ...)")
        if not 2 <= self.strength <= self.levels:
            raise ValueError("strength must be between 2 and the number of levels")
        if self.n_factors > self.levels + 1:
            raise ValueError(f"at most levels + 1 = {self.levels + 1} factors")

    def array(self):
        """The integer array, shape ``(s^t, n_factors)`` with levels ``0..s-1``."""
        s, t = self.levels, self.strength
        add, mul = _gf_tables(s)
        rows = []
        for coeffs in np.ndindex(*([s] * t)):
            row = []
            for x in range(s):
                v, xp = 0, 1
                for c in coeffs:           # c_0 + c_1 x + ... + c_{t-1} x^{t-1}
                    v = add[v, mul[c, xp]]
                    xp = mul[xp, x]
                row.append(int(v))
            row.append(int(coeffs[-1]))    # the "point at infinity" column
            rows.append(row)
        return np.array(rows, dtype=int)[:, :self.n_factors]

    def _generate(self, rng):
        A = self.array()
        s = self.levels
        N = len(A)
        if self.lhs:
            U = np.empty(A.shape)
            per = N // s
            for j in range(A.shape[1]):
                for lv in range(s):
                    rows = np.where(A[:, j] == lv)[0]
                    U[rows, j] = lv * per + rng.permutation(per)
            U = (U + 0.5) / N
        else:
            U = (A + 0.5) / s
        props = {"strength": self.strength, "index": N // s ** self.strength,
                 "n_levels": s, "oa_lhs": self.lhs, "is_lhs": bool(_is_lhs(U))}
        d = _unit_design(U, "OA-based Latin hypercube" if self.lhs else self.kind,
                         self.bounds, props)
        self.array_ = A
        return d
