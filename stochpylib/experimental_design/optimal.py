"""Exact optimal designs by point exchange: D-, A-, G-, I- and T-optimality, and
(pseudo-)Bayesian D-optimality for linear and nonlinear models.

Every class subclasses :class:`~stochpylib.experimental_design.OptimalDesign`, which owns
the multi-start exchange over a candidate set; each one only defines its criterion.
Efficiencies in ``design_.properties`` are fractions in (0, 1]: ``D_efficiency`` is
``det(F'F)^(1/p)/n``, ``A_efficiency`` ``p/trace(n (F'F)^-1)`` and ``G_efficiency``
``p / (n max_x f(x)'(F'F)^-1 f(x))`` over the candidate set (1 at a G-optimal design, by
the Kiefer-Wolfowitz equivalence theorem).
"""

import numpy as np

from stochpylib.experimental_design._base import OptimalDesign, _batched_inverse
from stochpylib.experimental_design._common import _model_matrix, _model_terms, _sobol_reference

__all__ = ["D_OptimalDesign", "A_OptimalDesign", "G_OptimalDesign", "I_OptimalDesign",
           "T_OptimalDesign", "BayesianDesign"]


class D_OptimalDesign(OptimalDesign):
    """D-optimal design: maximizes ``det(F'F)``, i.e. minimizes the volume of the
    parameter confidence ellipsoid."""

    kind = "D-optimal"
    criterion_name = "-log det(F'F)"

    def _score(self, M):
        _, logdet, singular = _batched_inverse(M)
        return np.where(singular.any(-1), np.inf, -logdet.mean(-1))


class A_OptimalDesign(OptimalDesign):
    """A-optimal design: minimizes ``trace((F'F)^-1)``, the average parameter variance."""

    kind = "A-optimal"
    criterion_name = "trace((F'F)^-1)"

    def _score(self, M):
        inv, _, singular = _batched_inverse(M)
        tr = np.trace(inv, axis1=-2, axis2=-1).mean(-1)
        return np.where(singular.any(-1), np.inf, tr)


class G_OptimalDesign(OptimalDesign):
    """G-optimal design: minimizes the maximum prediction variance
    ``f(x)'(F'F)^-1 f(x)`` over the candidate set."""

    kind = "G-optimal"
    criterion_name = "max f'(F'F)^-1 f"

    def _score(self, M):
        inv, _, singular = _batched_inverse(M[..., 0, :, :])
        Fc = self._G_cand[0]
        var = np.einsum("cp,...pq,cq->...c", Fc, inv, Fc)
        return np.where(singular, np.inf, var.max(-1))


class I_OptimalDesign(OptimalDesign):
    """I-optimal (IV-optimal) design: minimizes the prediction variance averaged over the
    region, ``trace(M_R (F'F)^-1)``.

    For polynomial models ``M_R`` is the exact moment matrix of the uniform distribution
    on ``[-1, 1]^k`` (``E[x^a] = 1/(a+1)`` for even ``a``); a callable model uses 4096
    Sobol points instead.
    """

    kind = "I-optimal"
    criterion_name = "trace(M_R (F'F)^-1)"

    def _moments(self, k):
        if callable(self.model):
            X = 2.0 * _sobol_reference(k, 4096, 0) - 1.0
            F, _ = _model_matrix(X, self.model)
            return F.T @ F / len(F)
        terms = np.array(_model_terms(k, self.model))
        p = len(terms)
        MR = np.empty((p, p))
        for a in range(p):
            for b in range(p):
                e = terms[a] + terms[b]
                MR[a, b] = np.prod(np.where(e % 2 == 0, 1.0 / (e + 1.0), 0.0))
        return MR

    def _features(self, C):
        self._MR = self._moments(C.shape[1])
        return super()._features(C)

    def _score(self, M):
        inv, _, singular = _batched_inverse(M[..., 0, :, :])
        val = np.einsum("pq,...qp->...", self._MR, inv)
        return np.where(singular, np.inf, val)


class T_OptimalDesign(OptimalDesign):
    """T-optimal design (Atkinson & Fedorov 1975) for discriminating between two models.

    ``true_model(X) -> (n,)`` is the mean response under the model assumed true (with its
    parameters fixed); the design maximizes the residual sum of squares left when the
    ``rival_model`` (a polynomial model name / term list) is least-squares fitted to it --
    the lack of fit the experiment can expose. ``model`` is unused here.
    """

    kind = "T-optimal"
    criterion_name = "-RSS(rival | true)"

    def __init__(self, n_runs, true_model, rival_model="linear", n_factors=None,
                 candidates=None, levels=3, bounds=None, n_starts=5, max_iter=100, tol=1e-10,
                 random_state=None):
        if not callable(true_model):
            raise ValueError("true_model must be a callable eta(X) -> (n,)")
        self.true_model = true_model
        self.rival_model = rival_model
        super().__init__(n_runs, n_factors=n_factors, model=rival_model, candidates=candidates,
                         levels=levels, bounds=bounds, n_starts=n_starts, max_iter=max_iter,
                         tol=tol, random_state=random_state)

    def _features(self, C):
        self._eta = np.asarray(self.true_model(C), dtype=float).ravel()
        F, _ = _model_matrix(C, self.rival_model)
        self._F2 = F
        return F[None, :, :]

    def _min_runs(self, p):
        return p + 1

    def _rss(self, idx):
        F, e = self._F2[idx], self._eta[idx]
        beta, *_ = np.linalg.lstsq(F, e, rcond=None)
        return float(np.sum((e - F @ beta) ** 2))

    def _design_score(self, G, idx):
        return -self._rss(idx)

    def _swap_scores(self, G, idx, i, M):
        out = np.empty(G.shape[1])
        trial = idx.copy()
        for j in range(G.shape[1]):
            trial[i] = j
            out[j] = -self._rss(trial)
        return out

    def _generate(self, rng):
        d = super()._generate(rng)
        d.properties["rss"] = -d.properties["criterion"]
        return d


class BayesianDesign(OptimalDesign):
    """Bayesian D-optimal design.

    - **Linear model** (``model`` a polynomial name / term list): maximizes
      ``log det(F'F + R)``, with ``R = prior_precision`` the prior precision of the
      coefficients (in units of the error variance). ``R = 0`` recovers D-optimality.
    - **Nonlinear model** (``model(X, theta) -> (n,)``): the pseudo-Bayesian criterion
      ``E_prior[log det(J(theta)'J(theta))]``, averaged over ``n_prior_samples`` draws of
      ``prior`` (a sequence of stochpylib distribution objects, one per parameter, or an
      ``(m, q)`` array of draws). ``J`` is the Jacobian of the mean in ``theta`` -- given
      via ``jacobian(X, theta)`` or by central finite differences. Candidates are then a
      ``levels`` grid on ``bounds`` in **natural** units, and the Design keeps them.
    """

    kind = "Bayesian D-optimal"
    criterion_name = "-E log det(information)"

    def __init__(self, n_runs, model="linear", prior_precision=None, prior=None,
                 n_prior_samples=64, jacobian=None, n_factors=None, candidates=None, levels=3,
                 bounds=None, n_starts=5, max_iter=100, tol=1e-10, random_state=None):
        self.prior_precision = prior_precision
        self.prior = prior
        self.n_prior_samples = int(n_prior_samples)
        self.jacobian = jacobian
        super().__init__(n_runs, n_factors=n_factors, model=model, candidates=candidates,
                         levels=levels, bounds=bounds, n_starts=n_starts, max_iter=max_iter,
                         tol=tol, random_state=random_state)

    def _nonlinear(self):
        return callable(self.model) and self.prior is not None

    def _natural_space(self):
        return self._nonlinear() and self.candidates is None

    def _draws(self, rng):
        prior = self.prior
        if not hasattr(prior[0], "rvs"):
            draws = np.asarray(prior, dtype=float)
            return draws[:, None] if draws.ndim == 1 else draws
        cols = [np.atleast_1d(np.asarray(d.rvs(self.n_prior_samples, random_state=rng),
                                         dtype=float)) for d in prior]
        return np.column_stack(cols)

    def _jac(self, C, theta):
        if self.jacobian is not None:
            return np.asarray(self.jacobian(C, theta), dtype=float)
        q = len(theta)
        J = np.empty((len(C), q))
        for r in range(q):
            h = 1e-6 * max(1.0, abs(theta[r]))
            tp, tm = theta.copy(), theta.copy()
            tp[r] += h
            tm[r] -= h
            J[:, r] = (np.asarray(self.model(C, tp), dtype=float)
                       - np.asarray(self.model(C, tm), dtype=float)) / (2 * h)
        return J

    def _generate(self, rng):
        self._rng_prior = rng
        return super()._generate(rng)

    def _features(self, C):
        if self._nonlinear():
            self._theta = self._draws(self._rng_prior)
            return np.stack([self._jac(C, th) for th in self._theta])
        if callable(self.model):
            raise ValueError("a callable model needs a prior (nonlinear Bayesian design)")
        G = super()._features(C)
        p = G.shape[2]
        R = np.zeros((p, p)) if self.prior_precision is None else np.asarray(
            self.prior_precision, dtype=float)
        if np.ndim(R) == 0:
            R = float(R) * np.eye(p)
        elif np.ndim(R) == 1:
            R = np.diag(R)
        if R.shape != (p, p):
            raise ValueError(f"prior_precision must be ({p}, {p})")
        self._R = R
        return G

    def _min_runs(self, p):
        if self._nonlinear():
            return p
        return 1 if np.any(np.linalg.eigvalsh(self._R) > 0) and np.all(
            np.linalg.eigvalsh(self._R) > 0) else p

    def _score(self, M):
        if self._nonlinear():
            _, logdet, singular = _batched_inverse(M)
            return np.where(singular.any(-1), np.inf, -logdet.mean(-1))
        _, logdet, singular = _batched_inverse(M + self._R)
        return np.where(singular.any(-1), np.inf, -logdet.mean(-1))

    def _efficiencies(self, G, idx):
        return {} if self._nonlinear() else super()._efficiencies(G, idx)
