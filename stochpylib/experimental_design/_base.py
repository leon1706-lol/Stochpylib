"""Generator base classes for :mod:`stochpylib.experimental_design`.

:class:`DesignGenerator` owns seed resolution and result storage for every design class;
subclasses implement ``_generate(rng) -> Design``. :class:`OptimalDesign` adds the
modified-Fedorov point-exchange engine shared by the D/A/G/I/T-optimal and Bayesian
designs, which differ only in the criterion they score.
"""

import itertools

import numpy as np

from stochpylib.experimental_design._common import (
    _factor_names,
    _model_matrix,
    _model_terms,
    _normalize_bounds,
    _rng,
    _term_name,
)
from stochpylib.experimental_design._design import Design

__all__ = ["DesignGenerator", "OptimalDesign"]


class DesignGenerator:
    """Base for every design class.

    ``generate(random_state=None)`` resolves the generator (the constructor's
    ``random_state`` unless overridden), calls the subclass hook ``_generate(rng)``, stores
    the resulting :class:`~stochpylib.experimental_design.Design` on ``self.design_`` and
    returns it. Deterministic designs simply ignore the generator.
    """

    kind = "design"

    def generate(self, random_state=None):
        seed = getattr(self, "random_state", None) if random_state is None else random_state
        design = self._generate(_rng(seed))
        self.design_ = design
        return design

    def _generate(self, rng):
        raise NotImplementedError

    def __repr__(self):
        if not hasattr(self, "design_"):
            return f"{type(self).__name__}(unfitted)"
        return (f"{type(self).__name__}(n_runs={self.design_.n_runs}, "
                f"n_factors={self.design_.n_factors})")


def _batched_inverse(M):
    """Eigen-based inverse of a stack of symmetric matrices plus a singularity mask."""
    w, V = np.linalg.eigh(M)
    top = np.maximum(w[..., -1:], 1e-300)
    singular = (w[..., 0] <= 1e-12 * top[..., 0]) | (w[..., -1] <= 0)
    w_safe = np.where(w > 0, w, 1.0)
    inv = (V / w_safe[..., None, :]) @ np.swapaxes(V, -1, -2)
    logdet = np.sum(np.log(w_safe), axis=-1)
    return inv, logdet, singular


class OptimalDesign(DesignGenerator):
    """Base for exact optimal designs found by point exchange over a candidate set.

    The candidate set is ``candidates`` (coded ``[-1, 1]`` units) or, by default, the full
    ``levels``-per-factor grid on ``[-1, 1]^k``. Each of ``n_starts`` random starts draws
    ``n_runs`` candidates with replacement and then sweeps the runs, replacing each by the
    candidate that most improves the criterion, until a sweep no longer helps (a local
    optimum -- hence the multi-start). Subclasses define ``_score(M)``: the criterion (lower
    is better, ``+inf`` when singular) for a stack of information matrices of shape
    ``(..., S, p, p)``, where ``S > 1`` only for pseudo-Bayesian designs (one information
    matrix per prior draw).
    """

    kind = "optimal design"
    criterion_name = "optimal"

    def __init__(self, n_runs, n_factors=None, model="linear", candidates=None, levels=3,
                 bounds=None, n_starts=5, max_iter=100, tol=1e-10, random_state=None):
        self.n_runs = int(n_runs)
        self.n_factors = n_factors
        self.model = model
        self.candidates = candidates
        self.levels = levels
        self.bounds = bounds
        self.n_starts = int(n_starts)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    # ------------------------------------------------------------------ candidate set
    def _natural_space(self):
        return False

    def _candidate_points(self):
        if self.candidates is not None:
            C = np.asarray(self.candidates, dtype=float)
            return C[:, None] if C.ndim == 1 else C
        k = self.n_factors
        if k is None:
            if self.bounds is not None and np.ndim(self.bounds) == 2:
                k = len(self.bounds)
            elif self.bounds is not None and np.size(self.bounds) == 2:
                k = 1
            else:
                raise ValueError(f"{type(self).__name__} needs n_factors or candidates")
        k = int(k)
        levels = [self.levels] * k if np.ndim(self.levels) == 0 else list(self.levels)
        if self._natural_space():
            b = _normalize_bounds(self.bounds, k)
            axes = [np.linspace(lo, hi, int(m)) for (lo, hi), m in zip(b, levels)]
        else:
            axes = [np.linspace(-1.0, 1.0, int(m)) for m in levels]
        return np.array(list(itertools.product(*axes)), dtype=float)

    def _features(self, C):
        """Candidate features of shape ``(S, m, p)``."""
        F, _ = _model_matrix(C, self.model)
        return F[None, :, :]

    def _term_names(self, k):
        if callable(self.model):
            return None
        return [_term_name(t, _factor_names(k)) for t in _model_terms(k, self.model)]

    # --------------------------------------------------------------------- criterion
    def _score(self, M):
        raise NotImplementedError

    def _info(self, G, idx):
        g = G[:, idx, :]
        return np.einsum("sip,siq->spq", g, g)

    def _swap_scores(self, G, idx, i, M):
        gi = G[:, idx[i], :]
        base = M - gi[:, :, None] * gi[:, None, :]
        Gc = np.swapaxes(G, 0, 1)                      # (m, S, p)
        stack = base[None] + Gc[:, :, :, None] * Gc[:, :, None, :]
        return self._score(stack)

    def _design_score(self, G, idx):
        return float(self._score(self._info(G, idx)[None])[0])

    # ---------------------------------------------------------------------- exchange
    def _generate(self, rng):
        C = self._candidate_points()
        G = self._features(C)
        self._G_cand = G
        m, p = G.shape[1], G.shape[2]
        if self.n_runs < self._min_runs(p):
            raise ValueError(f"n_runs={self.n_runs} cannot support a {p}-parameter model")
        best_idx, best_val, total_iter = None, np.inf, 0
        for _ in range(max(1, self.n_starts)):
            idx = rng.integers(0, m, self.n_runs)
            val = self._design_score(G, idx)
            for _retry in range(50):
                if np.isfinite(val):
                    break
                idx = rng.integers(0, m, self.n_runs)
                val = self._design_score(G, idx)
            for it in range(self.max_iter):
                total_iter += 1
                improved = False
                for i in range(self.n_runs):
                    M = self._info(G, idx)
                    scores = self._swap_scores(G, idx, i, M)
                    j = int(np.argmin(scores))
                    thresh = self.tol * max(1.0, abs(val)) if np.isfinite(val) else 0.0
                    if scores[j] < val - thresh:
                        idx[i] = j
                        val = float(scores[j])
                        improved = True
                if not improved:
                    break
            if val < best_val:
                best_val, best_idx = val, idx.copy()
        if best_idx is None or not np.isfinite(best_val):
            raise ValueError("no nonsingular design found; enlarge n_runs or the candidate set")
        pts = C[np.sort(best_idx)]
        props = {"criterion": best_val, "criterion_name": self.criterion_name,
                 "n_iter": total_iter, "n_candidates": m, "n_starts": self.n_starts}
        props.update(self._efficiencies(G, np.sort(best_idx)))
        k = C.shape[1]
        terms = self._term_names(k)
        if terms is not None:
            props["model_terms"] = terms
        space = "natural" if self._natural_space() else "coded"
        return Design(pts, space=space, kind=self.kind, bounds=self.bounds, properties=props)

    def _min_runs(self, p):
        return p

    def _efficiencies(self, G, idx):
        if G.shape[0] != 1:
            return {}
        F = G[0, idx, :]
        n, p = F.shape
        inv, logdet, singular = _batched_inverse((F.T @ F)[None])
        if singular[0]:
            return {"D_efficiency": 0.0, "A_efficiency": 0.0, "G_efficiency": 0.0}
        inv = inv[0]
        d_max = float(np.max(np.einsum("mp,pq,mq->m", G[0], inv, G[0])))
        return {"D_efficiency": float(np.exp(logdet[0] / p) / n),
                "A_efficiency": float(p / (n * np.trace(inv))),
                "G_efficiency": float(p / (n * d_max))}
