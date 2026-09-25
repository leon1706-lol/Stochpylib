"""Response surfaces and surrogate models: second-order response-surface methodology with
canonical analysis and its ANOVA, polynomial chaos expansions with analytic Sobol indices,
and universal kriging. Every surrogate subclasses :class:`MetaModel` (``fit(X, y)``
returning ``self``, ``predict``, ``score``, ``cross_validate``); instantiated directly,
``MetaModel`` picks the best of them by k-fold cross-validation.
"""

import copy
import itertools
import math

import numpy as np

from stochpylib.experimental_design._common import (
    _as_2d,
    _model_matrix,
    _model_terms,
    _normalize_bounds,
    _rng,
)

__all__ = ["MetaModel", "ResponseSurface", "PolynomialChaos", "KrigingSurrogate",
           "RSM_ANOVA"]


def _as_y(y, n):
    y = np.asarray(y, dtype=float).ravel()
    if y.size != n:
        raise ValueError(f"y must have {n} values, one per run")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")
    return y


def _kfold(n, k, rng):
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


# ---------------------------------------------------------------------------- MetaModel

class MetaModel:
    """Base class of every surrogate, and -- instantiated directly -- a selector.

    Subclasses implement ``_fit(X, y)`` and ``_predict(X, return_std)``. Used on its own,
    ``MetaModel(candidates=None, cv=5).fit(X, y)`` scores each candidate surrogate by k-fold
    cross-validated RMSE (default candidates: first- and second-order
    :class:`ResponseSurface`, a degree-3 :class:`PolynomialChaos` on the data's range, and
    :class:`KrigingSurrogate`), refits the winner on all the data as ``best_`` and
    delegates ``predict`` to it; ``scores_`` holds every candidate's CV RMSE.
    """

    kind = "metamodel"

    def __init__(self, candidates=None, cv=5, random_state=None):
        self.candidates = candidates
        self.cv = int(cv)
        self.random_state = random_state

    def fit(self, X, y):
        X = _as_2d(X)
        y = _as_y(y, len(X))
        self._fit(X, y)
        self.X_train_, self.y_train_ = X, y
        return self

    def predict(self, X, return_std=False):
        if not hasattr(self, "X_train_"):
            raise RuntimeError("fit() must be called first")
        return self._predict(_as_2d(X), return_std)

    def __call__(self, X):
        return self.predict(X)

    def score(self, X, y):
        """Coefficient of determination R^2 of the predictions on ``(X, y)``."""
        X = _as_2d(X)
        y = _as_y(y, len(X))
        resid = y - np.asarray(self.predict(X), dtype=float)
        sst = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else float("nan")

    def cross_validate(self, X, y, k=5, random_state=None):
        """k-fold CV of a fresh copy: ``{"rmse", "q2", "fold_rmse"}``."""
        X = _as_2d(X)
        y = _as_y(y, len(X))
        k = min(int(k), len(X))
        preds = np.empty(len(y))
        fold_rmse = []
        for test in _kfold(len(X), k, _rng(random_state)):
            train = np.setdiff1d(np.arange(len(X)), test)
            model = self._fresh().fit(X[train], y[train])
            preds[test] = np.asarray(model.predict(X[test]), dtype=float)
            fold_rmse.append(float(np.sqrt(np.mean((preds[test] - y[test]) ** 2))))
        sst = float(np.sum((y - y.mean()) ** 2))
        return {"rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
                "q2": 1.0 - float(np.sum((preds - y) ** 2)) / sst if sst > 0 else float("nan"),
                "fold_rmse": fold_rmse}

    def _fresh(self):
        clone = copy.copy(self)
        for attr in [a for a in vars(clone) if a.endswith("_") and not a.startswith("_")]:
            delattr(clone, attr)
        clone._reset()
        return clone

    def _reset(self):
        pass

    # ----------------------------------------------------------------- selector mode
    def _default_candidates(self, X):
        bounds = np.column_stack([X.min(0), X.max(0)])
        bounds[:, 1] = np.where(bounds[:, 1] > bounds[:, 0], bounds[:, 1], bounds[:, 0] + 1.0)
        return {"response_surface_1": ResponseSurface(order=1),
                "response_surface_2": ResponseSurface(order=2),
                "polynomial_chaos_3": PolynomialChaos(degree=3, bounds=bounds),
                "kriging": KrigingSurrogate()}

    def _fit(self, X, y):
        if type(self) is not MetaModel:
            raise NotImplementedError
        cands = self.candidates if self.candidates is not None else self._default_candidates(X)
        if not isinstance(cands, dict):
            cands = {f"{type(c).__name__}_{i}": c for i, c in enumerate(cands)}
        scores = {}
        for name, model in cands.items():
            try:
                scores[name] = model.cross_validate(X, y, k=self.cv,
                                                    random_state=self.random_state)["rmse"]
            except (ValueError, np.linalg.LinAlgError):
                scores[name] = np.inf
        best = min(scores, key=scores.get)
        if not np.isfinite(scores[best]):
            raise ValueError("no candidate surrogate could be fitted to these data")
        self.scores_ = scores
        self.best_name_ = best
        self.best_ = cands[best]._fresh().fit(X, y)

    def _predict(self, X, return_std):
        if type(self) is not MetaModel:
            raise NotImplementedError
        return self.best_.predict(X, return_std=return_std)

    def __repr__(self):
        state = "fitted" if hasattr(self, "X_train_") else "unfitted"
        return f"{type(self).__name__}({state})"


# ------------------------------------------------------------------ response surface

_ORDER_MODEL = {1: "linear", 2: "quadratic", 3: "cubic"}


class ResponseSurface(MetaModel):
    """Polynomial response surface fitted by ordinary least squares.

    ``order`` 1/2/3 selects the linear/quadratic/cubic model (``model=`` overrides it with
    any model name or exponent-tuple list). With ``bounds``, inputs are coded to
    ``[-1, 1]`` before fitting and every point returned (stationary point, optimum, ascent
    path) is decoded back to natural units. The fit is
    :func:`stochpylib.statistics.linear_regression` -- ``regression_`` is its
    ``RegressionResult`` -- so coefficients arrive with standard errors and p-values.
    """

    kind = "response surface"

    def __init__(self, order=2, model=None, bounds=None):
        self.order = int(order)
        if model is None and self.order not in _ORDER_MODEL:
            raise ValueError("order must be 1, 2 or 3 (or pass model=)")
        self.model = model if model is not None else _ORDER_MODEL[self.order]
        if callable(self.model):
            raise ValueError("ResponseSurface needs a polynomial model, not a callable")
        self.bounds = bounds

    def _code(self, X):
        if self.bounds is None:
            return X
        b = _normalize_bounds(self.bounds, X.shape[1])
        return 2.0 * (X - b[:, 0]) / (b[:, 1] - b[:, 0]) - 1.0

    def _decode(self, Z):
        if self.bounds is None:
            return Z
        b = _normalize_bounds(self.bounds, Z.shape[-1])
        return b[:, 0] + (Z + 1.0) / 2.0 * (b[:, 1] - b[:, 0])

    def _fit(self, X, y):
        from stochpylib.statistics import linear_regression

        Z = self._code(X)
        k = Z.shape[1]
        self.terms_exp_ = _model_terms(k, self.model)
        if self.terms_exp_[0] != (0,) * k:
            raise ValueError("the model must start with the intercept term")
        F, names = _model_matrix(Z, self.model)
        if len(Z) <= F.shape[1]:
            raise ValueError(f"{F.shape[1]} coefficients need more than {len(Z)} runs")
        reg = linear_regression(F[:, 1:], y)
        self.regression_ = reg
        self.coef_ = np.asarray(reg.coef_, dtype=float)
        self.terms_ = names
        self.n_factors_ = k

    def _predict(self, X, return_std):
        F, _ = _model_matrix(self._code(X), self.model)
        mu = F @ self.coef_
        if not return_std:
            return mu
        cov = np.asarray(self.regression_.cov_params_, dtype=float)
        return mu, np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", F, cov, F), 0.0))

    def _linear_and_quadratic(self):
        k = self.n_factors_
        b, B = np.zeros(k), np.zeros((k, k))
        for c, e in zip(self.coef_, self.terms_exp_):
            e = np.asarray(e)
            if e.sum() == 1:
                b[int(np.argmax(e))] = c
            elif e.sum() == 2:
                idx = np.where(e > 0)[0]
                if len(idx) == 1:
                    B[idx[0], idx[0]] = c
                else:
                    B[idx[0], idx[1]] = B[idx[1], idx[0]] = c / 2.0
            elif e.sum() > 2:
                raise ValueError("stationary/canonical analysis needs a second-order model")
        return b, B

    def stationary_point(self):
        """``x_s = -B^-1 b / 2`` of the fitted second-order surface (natural units)."""
        b, B = self._linear_and_quadratic()
        if not np.any(B):
            raise ValueError("a first-order surface has no stationary point")
        xs = -0.5 * np.linalg.solve(B, b)
        return self._decode(xs[None, :])[0]

    def canonical_analysis(self):
        """Eigen-analysis of the quadratic part: ``stationary_point``, ``eigenvalues``,
        ``eigenvectors``, ``predicted`` value there, and the surface's ``nature``
        (``"maximum"``, ``"minimum"``, ``"saddle"`` or ``"ridge"``)."""
        b, B = self._linear_and_quadratic()
        lam, vec = np.linalg.eigh(B)
        scale = max(np.max(np.abs(lam)), 1e-300)
        if np.any(np.abs(lam) < 1e-8 * scale):
            nature = "ridge"
        elif np.all(lam < 0):
            nature = "maximum"
        elif np.all(lam > 0):
            nature = "minimum"
        else:
            nature = "saddle"
        xs = self.stationary_point() if nature != "ridge" else None
        pred = float(self.predict(xs[None, :])[0]) if xs is not None else None
        return {"stationary_point": xs, "eigenvalues": lam, "eigenvectors": vec,
                "predicted": pred, "nature": nature}

    def steepest_ascent(self, n_steps=10, step=0.5, maximize=True):
        """Path of steepest ascent (descent with ``maximize=False``) from the design centre,
        in coded steps of length ``step``: ``{"path", "predicted"}``."""
        b, B = self._linear_and_quadratic()
        sign = 1.0 if maximize else -1.0
        z = np.zeros(self.n_factors_)
        path = [z.copy()]
        for _ in range(int(n_steps)):
            g = b + 2.0 * B @ z
            norm = np.linalg.norm(g)
            if norm < 1e-14:
                break
            z = z + sign * step * g / norm
            path.append(z.copy())
        P = self._decode(np.array(path))
        return {"path": P, "predicted": np.asarray(self.predict(P), dtype=float)}

    def optimize(self, bounds=None, maximize=False, random_state=None):
        """Optimum of the fitted surface inside a box: ``{"x", "value"}``.

        The box is ``bounds``, else the model's own ``bounds``, else the training range.
        Searched by :class:`stochpylib.optimization.DifferentialEvolution`; the stationary
        point also competes when it lies inside the box.
        """
        from stochpylib.optimization import DifferentialEvolution

        if bounds is not None:
            box = _normalize_bounds(bounds, self.n_factors_)
        elif self.bounds is not None:
            box = _normalize_bounds(self.bounds, self.n_factors_)
        else:
            box = np.column_stack([self.X_train_.min(0), self.X_train_.max(0)])
        sign = -1.0 if maximize else 1.0
        fun = lambda x: sign * float(self.predict(np.asarray(x)[None, :])[0])
        de = DifferentialEvolution(bounds=[tuple(r) for r in box], n_iter=200,
                                   random_state=random_state).minimize(fun, box.mean(1))
        best_x, best_v = np.asarray(de.x_, dtype=float), float(de.fun_)
        try:
            xs = self.stationary_point()
            if np.all(xs >= box[:, 0] - 1e-12) and np.all(xs <= box[:, 1] + 1e-12):
                vs = fun(xs)
                if vs <= best_v:
                    best_x, best_v = xs, vs
        except (ValueError, np.linalg.LinAlgError):
            pass
        return {"x": best_x, "value": sign * best_v}


# ------------------------------------------------------------------------ RSM ANOVA

class RSM_ANOVA:
    """ANOVA of a response-surface fit.

    ``fit(X, y)`` splits the regression sum of squares **sequentially** into its
    ``linear``, ``interaction`` (two-factor products) and ``quadratic`` (pure squares)
    groups, splits the residual into ``lack of fit`` and ``pure error`` whenever runs are
    replicated (identical rows of X, e.g. centre points), and reports ``r2_``,
    ``adj_r2_`` and ``pred_r2_`` (``1 - PRESS/SST``). ``table_`` is a
    ``statistics.TestResult`` whose statistic is the overall regression F test;
    ``lack_of_fit_`` is the lack-of-fit F test (``None`` without replicates).
    """

    def __init__(self, order=2, bounds=None):
        self.order = int(order)
        if self.order not in (1, 2):
            raise ValueError("RSM_ANOVA supports first- and second-order models")
        self.bounds = bounds

    @staticmethod
    def _rss(F, y):
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ beta
        return float(r @ r)

    def fit(self, X, y):
        from stochpylib.statistics._common import _f_sf
        from stochpylib.statistics._result import TestResult

        X = _as_2d(X)
        y = _as_y(y, len(X))
        if self.bounds is not None:
            b = _normalize_bounds(self.bounds, X.shape[1])
            X = 2.0 * (X - b[:, 0]) / (b[:, 1] - b[:, 0]) - 1.0
        n, k = X.shape
        cols = [np.ones(n)]
        groups = [("linear", [X[:, i] for i in range(k)])]
        if self.order == 2:
            groups.append(("interaction", [X[:, i] * X[:, j]
                                           for i, j in itertools.combinations(range(k), 2)]))
            groups.append(("quadratic", [X[:, i] ** 2 for i in range(k)]))
        rows = []
        sst = float(np.sum((y - y.mean()) ** 2))
        prev = sst
        for name, g in groups:
            if not g:
                continue
            cols += g
            cur = self._rss(np.column_stack(cols), y)
            rows.append({"source": name, "ss": prev - cur, "df": len(g)})
            prev = cur
        F = np.column_stack(cols)
        p = F.shape[1]
        df_res = n - p
        if df_res <= 0:
            raise ValueError(f"{p} coefficients need more than {n} runs")
        rss = prev
        ms_res = rss / df_res
        for r in rows:
            r["ms"] = r["ss"] / r["df"]
            r["F"] = r["ms"] / ms_res if ms_res > 0 else None
            r["p"] = float(_f_sf(r["F"], r["df"], df_res)) if r["F"] is not None else None
        rows.append({"source": "residual", "ss": rss, "df": df_res, "ms": ms_res,
                     "F": None, "p": None})
        # pure error from replicated runs
        keys = np.round(X, 12)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        ss_pe = float(sum(np.sum((y[inv == g] - y[inv == g].mean()) ** 2)
                          for g in np.unique(inv)))
        df_pe = n - len(np.unique(inv))
        self.lack_of_fit_ = None
        if df_pe > 0 and df_res - df_pe > 0:
            ss_lof, df_lof = rss - ss_pe, df_res - df_pe
            ms_lof, ms_pe = ss_lof / df_lof, ss_pe / df_pe
            f_lof = ms_lof / ms_pe if ms_pe > 0 else None
            p_lof = float(_f_sf(f_lof, df_lof, df_pe)) if f_lof is not None else None
            rows.append({"source": "lack of fit", "ss": ss_lof, "df": df_lof, "ms": ms_lof,
                         "F": f_lof, "p": p_lof})
            rows.append({"source": "pure error", "ss": ss_pe, "df": df_pe, "ms": ms_pe,
                         "F": None, "p": None})
            self.lack_of_fit_ = TestResult(f_lof, p_lof, (df_lof, df_pe),
                                           "the model fits (no lack of fit)",
                                           "lack-of-fit F test")
        rows.append({"source": "total", "ss": sst, "df": n - 1, "ms": None, "F": None,
                     "p": None})
        ss_reg = sst - rss
        f_reg = (ss_reg / (p - 1)) / ms_res if ms_res > 0 else None
        p_reg = float(_f_sf(f_reg, p - 1, df_res)) if f_reg is not None else None
        self.table_ = TestResult(f_reg, p_reg, (p - 1, df_res), "no regression effect",
                                 "response-surface ANOVA", table=rows)
        # PRESS through the hat diagonal: the leave-one-out residual is e_i / (1 - h_ii)
        H_diag = np.einsum("ij,ji->i", F, np.linalg.pinv(F))
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        e = y - F @ beta
        press = float(np.sum((e / np.maximum(1.0 - H_diag, 1e-12)) ** 2))
        self.r2_ = 1.0 - rss / sst
        self.adj_r2_ = 1.0 - (rss / df_res) / (sst / (n - 1))
        self.pred_r2_ = 1.0 - press / sst
        self.press_ = press
        self.coef_ = beta
        return self


# ------------------------------------------------------------------ polynomial chaos

def _legendre(z, p):
    """Orthonormal Legendre polynomials (uniform density on [-1, 1]) up to degree p."""
    out = np.empty((p + 1,) + z.shape)
    out[0] = 1.0
    if p >= 1:
        out[1] = z
    for n in range(1, p):
        out[n + 1] = ((2 * n + 1) * z * out[n] - n * out[n - 1]) / (n + 1)
    return out * np.sqrt(2.0 * np.arange(p + 1) + 1.0).reshape((-1,) + (1,) * z.ndim)


def _hermite(z, p):
    """Orthonormal probabilists' Hermite polynomials He_n / sqrt(n!) up to degree p."""
    out = np.empty((p + 1,) + z.shape)
    out[0] = 1.0
    if p >= 1:
        out[1] = z
    for n in range(1, p):
        out[n + 1] = z * out[n] - n * out[n - 1]
    norms = np.sqrt([math.factorial(n) for n in range(p + 1)])
    return out / norms.reshape((-1,) + (1,) * z.ndim)


class PolynomialChaos(MetaModel):
    """Polynomial chaos expansion ``y ~ sum_a c_a Psi_a(x)`` in an orthonormal basis.

    Input marginals come from ``distributions`` (stochpylib distribution objects, one per
    input) or ``bounds`` (independent uniforms). A ``Uniform`` input gets orthonormal
    Legendre polynomials, a ``Normal`` input normalized Hermite polynomials (the Wiener-Askey
    scheme); any other distribution is mapped to a standard normal through
    ``Phi^-1(F(x))`` and expanded in Hermite polynomials. Without either, ``fit`` uses
    uniforms on the training range. ``truncation="total"`` keeps ``|a|_1 <= degree``;
    ``"hyperbolic"`` keeps ``sum a_i^q <= degree^q``.

    Because the basis is orthonormal, the moments and Sobol indices are read off the
    coefficients: ``mean_ = c_0``, ``var_ = sum_{a != 0} c_a^2``, and ``sobol_first_`` /
    ``sobol_total_`` (Sudret 2008).
    """

    kind = "polynomial chaos"

    def __init__(self, degree=3, distributions=None, bounds=None, truncation="total", q=1.0):
        self.degree = int(degree)
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if truncation not in ("total", "hyperbolic"):
            raise ValueError("truncation must be 'total' or 'hyperbolic'")
        self.distributions = distributions
        self.bounds = bounds
        self.truncation = truncation
        self.q = float(q)

    # ---------------------------------------------------------------------- setup
    def _marginals(self, k, X=None):
        from stochpylib.distributions import Normal, Uniform

        if self.distributions is not None:
            dists = list(self.distributions)
            if len(dists) != k:
                raise ValueError(f"need {k} distributions, got {len(dists)}")
            out = []
            for d in dists:
                if isinstance(d, Uniform):
                    out.append(("legendre", ("uniform", d.a, d.b)))
                elif isinstance(d, Normal):
                    out.append(("hermite", ("normal", d.mu, d.sigma)))
                else:
                    out.append(("hermite", ("transform", d)))
            return out
        if self.bounds is not None:
            b = _normalize_bounds(self.bounds, k)
        elif X is not None:
            b = np.column_stack([X.min(0), X.max(0)])
        else:
            raise ValueError("PolynomialChaos needs distributions or bounds")
        return [("legendre", ("uniform", lo, hi)) for lo, hi in b]

    @staticmethod
    def _standardize(x, spec):
        from stochpylib.statistics._common import _norm_ppf

        kind = spec[0]
        if kind == "uniform":
            return 2.0 * (x - spec[1]) / (spec[2] - spec[1]) - 1.0
        if kind == "normal":
            return (x - spec[1]) / spec[2]
        u = np.clip(np.asarray(spec[1].cdf(x), dtype=float), 1e-15, 1 - 1e-15)
        return _norm_ppf(u)

    @staticmethod
    def _unstandardize(z, spec):
        from stochpylib.statistics._common import _norm_cdf

        kind = spec[0]
        if kind == "uniform":
            return spec[1] + (z + 1.0) / 2.0 * (spec[2] - spec[1])
        if kind == "normal":
            return spec[1] + spec[2] * z
        u = np.clip(_norm_cdf(z), 1e-15, 1 - 1e-15)
        return np.asarray(spec[1].ppf(u), dtype=float)

    def _multi_indices(self, k):
        p = self.degree
        idx = [a for total in range(p + 1)
               for a in self._compositions(total, k)]
        if self.truncation == "hyperbolic":
            idx = [a for a in idx if np.sum(np.asarray(a, dtype=float) ** self.q)
                   <= p ** self.q + 1e-12]
        return np.array(idx, dtype=int)

    @staticmethod
    def _compositions(total, k):
        if k == 1:
            return [(total,)]
        out = []
        for first in range(total, -1, -1):
            out += [(first,) + rest for rest in PolynomialChaos._compositions(total - first,
                                                                              k - 1)]
        return out

    def _basis(self, Z):
        per_dim = [(_legendre if kind == "legendre" else _hermite)(Z[:, i], self.degree)
                   for i, (kind, _) in enumerate(self._marg)]
        Psi = np.ones((len(Z), len(self.multi_indices_)))
        for i in range(Z.shape[1]):
            Psi *= per_dim[i][self.multi_indices_[:, i]].T
        return Psi

    def _to_z(self, X):
        return np.column_stack([self._standardize(X[:, i], spec)
                                for i, (_, spec) in enumerate(self._marg)])

    def _prepare(self, k, X=None):
        self._marg = self._marginals(k, X)
        self.multi_indices_ = self._multi_indices(k)
        self.n_factors_ = k

    # ------------------------------------------------------------------------ fitting
    def _fit(self, X, y):
        self._prepare(X.shape[1], X)
        Psi = self._basis(self._to_z(X))
        P = Psi.shape[1]
        if len(X) < P:
            raise ValueError(f"a degree-{self.degree} expansion has {P} terms; "
                             f"{len(X)} runs are too few")
        coef, *_ = np.linalg.lstsq(Psi, y, rcond=None)
        e = y - Psi @ coef
        h = np.einsum("ij,ji->i", Psi, np.linalg.pinv(Psi))
        var_y = float(np.var(y)) or 1.0
        self.loo_error_ = float(np.mean((e / np.maximum(1.0 - h, 1e-12)) ** 2) / var_y)
        self._set_coef(coef)

    def fit_function(self, fn, method="quadrature", n_samples=None, random_state=None):
        """Fit to a vectorized model ``fn(X) -> (n,)`` directly.

        ``method="quadrature"`` projects onto the basis with a tensor Gauss rule
        (``degree + 1`` nodes per input: Gauss-Legendre / probabilists' Gauss-Hermite from
        :mod:`stochpylib.numerical_methods`) -- exact for a polynomial model of degree <=
        ``degree + 1`` per input. ``method="regression"`` fits least squares on an
        ``n_samples`` Latin hypercube (default three times the number of terms).
        """
        from stochpylib.numerical_methods import GaussHermite, GaussLegendre

        if self.distributions is not None:
            k = len(self.distributions)
        elif self.bounds is not None:
            k = np.shape(self.bounds)[0] if np.ndim(self.bounds) == 2 else 1
        else:
            raise ValueError("fit_function needs distributions or bounds")
        self._prepare(k)
        if method == "quadrature":
            nodes, weights = [], []
            for kind, _ in self._marg:
                if kind == "legendre":
                    g = GaussLegendre(self.degree + 1)
                    nodes.append(g.nodes_)
                    weights.append(g.weights_ / 2.0)
                else:
                    g = GaussHermite(self.degree + 1, kind="probabilists")
                    nodes.append(g.nodes_)
                    weights.append(g.weights_)
            Z = np.array(list(itertools.product(*nodes)))
            W = np.prod(np.array(list(itertools.product(*weights))), axis=1)
            X = np.column_stack([self._unstandardize(Z[:, i], spec)
                                 for i, (_, spec) in enumerate(self._marg)])
            y = np.asarray(fn(X), dtype=float).ravel()
            Psi = self._basis(Z)
            self._set_coef(Psi.T @ (W * y))
            self.loo_error_ = None
            self.X_train_, self.y_train_ = X, y
            return self
        if method != "regression":
            raise ValueError("method must be 'quadrature' or 'regression'")
        from stochpylib.montecarlo import LatinHypercubeSampling

        n = int(n_samples) if n_samples is not None else 3 * len(self.multi_indices_)
        U = LatinHypercubeSampling(dim=k, n=n, random_state=_rng(random_state)).generate()
        U = np.clip(U, 1e-12, 1 - 1e-12)
        from stochpylib.statistics._common import _norm_ppf

        cols = []
        for i, (kind, spec) in enumerate(self._marg):
            z = 2.0 * U[:, i] - 1.0 if kind == "legendre" else _norm_ppf(U[:, i])
            cols.append(self._unstandardize(z, spec))
        X = np.column_stack(cols)
        return self.fit(X, np.asarray(fn(X), dtype=float).ravel())

    def _set_coef(self, coef):
        coef = np.asarray(coef, dtype=float)
        self.coef_ = coef
        self.mean_ = float(coef[0])
        self.var_ = float(np.sum(coef[1:] ** 2))
        k = self.multi_indices_.shape[1]
        V = self.var_ if self.var_ > 0 else 1.0
        A = self.multi_indices_
        first, total = np.zeros(k), np.zeros(k)
        for i in range(k):
            only_i = (A[:, i] > 0) & (np.sum(A > 0, axis=1) == 1)
            first[i] = np.sum(coef[only_i] ** 2) / V
            total[i] = np.sum(coef[A[:, i] > 0] ** 2) / V
        self.sobol_first_, self.sobol_total_ = first, total

    def _predict(self, X, return_std):
        if return_std:
            raise ValueError("PolynomialChaos has no predictive standard deviation")
        return self._basis(self._to_z(X)) @ self.coef_

    def fit(self, X, y):
        X = _as_2d(X)
        y = _as_y(y, len(X))
        self._fit(X, y)
        self.X_train_, self.y_train_ = X, y
        return self


# ---------------------------------------------------------------------------- kriging

_TRENDS = {"none": None, "constant": [0], "linear": "linear", "quadratic": "quadratic"}


class KrigingSurrogate(MetaModel):
    """Universal kriging: a Gaussian-process residual around a polynomial trend.

    Inputs are rescaled to the unit cube and responses standardized (``normalize=True``).
    The trend (``"none"``, ``"constant"`` -- ordinary kriging --, ``"linear"`` or
    ``"quadratic"``) is estimated by generalized least squares,
    ``beta = (F'K^-1 F)^-1 F'K^-1 y``, and the predictive variance carries the trend's
    estimation uncertainty, ``k(x,x) - k'K^-1 k + u'(F'K^-1 F)^-1 u``. The covariance is a
    :mod:`stochpylib.gaussian_processes` kernel -- by default an ARD Matern-5/2 -- whose
    hyperparameters (``optimize=True``) are fitted by
    :func:`~stochpylib.gaussian_processes.optimize_hyperparams` on the OLS-detrended data.
    """

    kind = "kriging"

    def __init__(self, kernel=None, trend="constant", noise=1e-8, optimize=True,
                 normalize=True):
        if trend not in _TRENDS:
            raise ValueError("trend must be 'none', 'constant', 'linear' or 'quadratic'")
        self.kernel = kernel
        self.trend = trend
        self.noise = float(noise)
        self.optimize = bool(optimize)
        self.normalize = bool(normalize)

    def _trend_matrix(self, Z):
        if self.trend == "none":
            return np.zeros((len(Z), 0))
        if self.trend == "constant":
            return np.ones((len(Z), 1))
        return _model_matrix(Z, self.trend)[0]

    def _scale(self, X):
        return (X - self._lo) / self._span if self.normalize else X

    def _fit(self, X, y):
        from stochpylib.gaussian_processes import (
            GPRegression, MaternKernel, cholesky_with_jitter, optimize_hyperparams,
        )

        n, d = X.shape
        self._lo = X.min(0)
        span = X.max(0) - self._lo
        self._span = np.where(span > 0, span, 1.0)
        if self.normalize:
            self._y_mu, self._y_sd = float(y.mean()), float(y.std()) or 1.0
        else:
            self._y_mu, self._y_sd = 0.0, 1.0
        Z = self._scale(X)
        ys = (y - self._y_mu) / self._y_sd
        kernel = (copy.deepcopy(self.kernel) if self.kernel is not None
                  else MaternKernel(nu=2.5, length_scale=0.5 * np.ones(d)))
        F = self._trend_matrix(Z)
        if self.optimize:
            resid = ys - F @ np.linalg.lstsq(F, ys, rcond=None)[0] if F.shape[1] else ys
            gp = GPRegression(kernel, noise=self.noise).fit(Z, resid)
            optimize_hyperparams(gp)
            kernel = gp.kernel
        K = kernel(Z) + self.noise * np.eye(n)
        L, jitter = cholesky_with_jitter(K)
        solve = lambda B: np.linalg.solve(L.T, np.linalg.solve(L, B))
        if F.shape[1]:
            KiF = solve(F)
            A = F.T @ KiF
            beta = np.linalg.solve(A, KiF.T @ ys)
            self._A_inv = np.linalg.inv(A)
            self._KiF = KiF
        else:
            beta = np.zeros(0)
        self._alpha = solve(ys - F @ beta)
        self._L, self._Z, self._solve = L, Z, solve
        self.kernel_ = kernel
        self.beta_ = beta
        self.jitter_ = jitter

    def _predict(self, X, return_std):
        Z = self._scale(X)
        k = self.kernel_(Z, self._Z)
        f = self._trend_matrix(Z)
        mu = f @ self.beta_ + k @ self._alpha
        mean = mu * self._y_sd + self._y_mu
        if not return_std:
            return mean
        v = np.linalg.solve(self._L, k.T)
        var = self.kernel_.diag(Z) - np.sum(v ** 2, axis=0)
        if f.shape[1]:
            u = f - k @ self._KiF
            var = var + np.einsum("ip,pq,iq->i", u, self._A_inv, u)
        return mean, np.sqrt(np.maximum(var, 0.0)) * self._y_sd

    def expected_improvement(self, X, y_best=None):
        """Expected improvement below ``y_best`` (default: the best training response)."""
        from stochpylib.statistics._common import _norm_cdf, _norm_pdf

        mu, sd = self.predict(X, return_std=True)
        best = float(np.min(self.y_train_)) if y_best is None else float(y_best)
        imp = best - mu
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sd > 1e-12, imp / sd, 0.0)
        ei = np.where(sd > 1e-12, imp * _norm_cdf(z) + sd * _norm_pdf(z),
                      np.maximum(imp, 0.0))
        return np.maximum(ei, 0.0)

    def sequential_design(self, fn, bounds, n_new, criterion="ei", n_candidates=500,
                          random_state=None):
        """Add ``n_new`` runs one at a time -- each where expected improvement
        (``criterion="ei"``) or predictive standard deviation (``"variance"``) is largest
        over ``n_candidates`` uniform candidates in ``bounds`` -- refitting after each."""
        if criterion not in ("ei", "variance"):
            raise ValueError("criterion must be 'ei' or 'variance'")
        rng = _rng(random_state)
        X, y = self.X_train_, self.y_train_
        b = _normalize_bounds(bounds, X.shape[1])
        for _ in range(int(n_new)):
            cand = b[:, 0] + rng.random((int(n_candidates), X.shape[1])) * (b[:, 1] - b[:, 0])
            if criterion == "ei":
                score = self.expected_improvement(cand)
            else:
                score = self.predict(cand, return_std=True)[1]
            x = cand[int(np.argmax(score))]
            X = np.vstack([X, x])
            y = np.append(y, float(np.asarray(fn(x[None, :]), dtype=float).ravel()[0]))
            self.fit(X, y)
        return self

