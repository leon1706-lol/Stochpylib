"""Module-level copula methods: dependence measures, dispatchers and helpers."""

import numpy as np

from stochpylib.copulas._utils import (
    kendall_tau_estimate, pseudo_obs, spearman_rho_estimate, as_u_matrix,
)

__all__ = [
    "CopulaFit", "CopulaSample", "kendall_tau", "spearman_rho",
    "tail_dependence", "copula_density", "conditional_copula",
]

_FAMILIES = {
    "gaussian": lambda: __import__(
        "stochpylib.copulas.elliptical", fromlist=["GaussianCopula"]
    ).GaussianCopula(),
    "t": lambda: __import__(
        "stochpylib.copulas.elliptical", fromlist=["StudentTCopula"]
    ).StudentTCopula(),
    "clayton": lambda: __import__(
        "stochpylib.copulas.archimedean", fromlist=["ClaytonCopula"]
    ).ClaytonCopula(),
    "gumbel": lambda: __import__(
        "stochpylib.copulas.archimedean", fromlist=["GumbelCopula"]
    ).GumbelCopula(),
    "frank": lambda: __import__(
        "stochpylib.copulas.archimedean", fromlist=["FrankCopula"]
    ).FrankCopula(),
    "joe": lambda: __import__(
        "stochpylib.copulas.archimedean", fromlist=["JoeCopula"]
    ).JoeCopula(),
    "amh": lambda: __import__(
        "stochpylib.copulas.archimedean", fromlist=["AliMikhailHaqCopula"]
    ).AliMikhailHaqCopula(),
    "plackett": lambda: __import__(
        "stochpylib.copulas.archimedean", fromlist=["PlackettCopula"]
    ).PlackettCopula(),
}


def kendall_tau(x, y=None):
    """Sample Kendall's tau-b of two columns (or of a (n, 2) array)."""
    if y is None:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("pass either two arrays or a single (n, 2) array")
        x, y = arr[:, 0], arr[:, 1]
    return float(kendall_tau_estimate(x, y))


def spearman_rho(x, y=None):
    """Sample Spearman's rho of two columns (or of a (n, 2) array)."""
    if y is None:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("pass either two arrays or a single (n, 2) array")
        x, y = arr[:, 0], arr[:, 1]
    return float(spearman_rho_estimate(x, y))


def tail_dependence(copula=None, data=None):
    """Tail-dependence coefficients.

    - ``tail_dependence(fitted_copula)`` -> analytic coefficients.
    - ``tail_dependence(data=data)`` -> empirical estimates from the standard
      threshold estimators ``lambda_L = C(u,u)/u`` and
      ``lambda_U = 2 - (1 - C(u,u))/(1-u)`` at ``u = 0.95``.
    - both -> analytic plus ``*_emp`` keys.
    """
    out = {}
    if copula is not None:
        out.update(copula.tail_dependence())
    if data is not None:
        u = pseudo_obs(data)
        uu = 0.95
        c_uu = float(np.mean(np.all(u <= uu, axis=1)))
        out["lower_emp"] = float(c_uu / uu)
        out["upper_emp"] = float(2.0 - (1.0 - c_uu) / (1.0 - uu))
    return out


def copula_density(copula, u):
    """Copula density evaluated on rows of ``u`` in the unit cube."""
    return np.asarray(copula.density(as_u_matrix(u)), dtype=float)


def conditional_copula(copula, w, given_v):
    """``P(U_1 <= w | U_2 = v)`` for a fitted bivariate copula."""
    if not hasattr(copula, "_h_u"):
        raise NotImplementedError(
            f"{type(copula).__name__} does not expose a conditional CDF")
    return np.asarray(copula._h_u(w, given_v), dtype=float)


class CopulaFit:
    """Fit every eligible family to bivariate data; rank by AIC::

        fit = CopulaFit().fit(data)
        fit.best_          # the winning fitted copula (fluent .sample etc.)
        fit.table_         # list of (family, aic) sorted best-first

    ``families`` defaults to all bivariate-capable parametric families.
    """

    def __init__(self, families=None, allow_rotations=False):
        self.families = tuple(families) if families else tuple(_FAMILIES)
        self.allow_rotations = bool(allow_rotations)
        self.best_ = None
        self.table_ = None

    def fit(self, data):
        from stochpylib.copulas.pair import _pair_loglik

        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("CopulaFit fits bivariate (n, 2) data")
        u = pseudo_obs(data)
        results = []
        for name in self.families:
            maker = _FAMILIES.get(name)
            if maker is None:
                continue
            cand = maker()
            try:
                cand.fit(u)
                ll = _pair_loglik(cand, u[:, 0], u[:, 1])
            except (ValueError, RuntimeError, NotImplementedError):
                continue
            n_params = getattr(cand, "_n_params", 1)
            aic = -2.0 * ll + 2.0 * n_params
            if np.isfinite(aic):
                results.append((name, float(aic), cand))
        if not results:
            raise RuntimeError("no family could be fitted to the data")
        results.sort(key=lambda r: r[1])
        self.table_ = [(name, aic) for name, aic, _ in results]
        self.best_ = results[0][2]
        self.best_name_ = results[0][0]
        return self


class CopulaSample:
    """Facade producing samples from any fitted bivariate copula object."""

    def __init__(self, copula):
        self.copula = copula

    def sample(self, n, random_state=None):
        return self.copula.sample(n, random_state=random_state)

    def cdf(self, u):
        return self.copula.cdf(u)
