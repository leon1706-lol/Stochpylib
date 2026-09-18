"""Shared result objects for :mod:`stochpylib.statistics`.

Mirrors the point-estimate + standard-error + confidence-interval convention used by
``stochpylib.montecarlo.MCResult`` and ``stochpylib.timeseries.TestResult`` (see
AGENTS.md Sec 3), but adapted to classical inference: a test needs a statistic and
p-value as well as (optionally) a point estimate of the effect; an estimator needs a
point estimate, standard error and confidence interval; a regression needs a whole
parameter vector's worth of both.
"""

from dataclasses import dataclass, field

import numpy as np

from stochpylib.statistics._common import _norm_ppf, _t_ppf


@dataclass
class TestResult:
    """Outcome of a classical hypothesis test.

    ``statistic``/``pvalue`` are always present. ``estimate``/``std_error``/``conf_int``
    are filled in for tests that also report an effect size (e.g. ``t_test``'s mean
    difference); ``table`` holds a structured breakdown for multi-row outputs (ANOVA,
    MANOVA, pairwise comparisons, multiple-testing adjustment) as a list of dicts.
    """

    statistic: float
    pvalue: float | None
    df: float | tuple | None = None
    null: str = ""
    method: str = ""
    alternative: str = "two-sided"
    estimate: float | np.ndarray | None = None
    std_error: float | None = None
    conf_int: tuple | None = None
    table: list | None = None
    extras: dict = field(default_factory=dict)

    def reject(self, alpha=0.05):
        """Whether the null is rejected at the given significance level."""
        return self.pvalue is not None and self.pvalue < alpha

    def __repr__(self):
        pv = f"{self.pvalue:.4g}" if self.pvalue is not None else "n/a"
        return f"TestResult(statistic={self.statistic:.4g}, pvalue={pv}, method={self.method!r})"


@dataclass
class EstimateResult:
    """A point estimate with standard error, sample size, and a confidence interval.

    ``confidence_interval(level)`` uses a Student-t interval when ``extras['df']`` is
    set, otherwise a normal-approximation interval. Used by every ``estimation.py``
    function, plus ``correlation()`` (``extras['pvalue']`` holds the significance test)
    and ``bayesian_estimator()`` (``extras['posterior']`` holds the posterior
    distribution object; the interval returned is the equal-tailed credible interval).
    """

    estimate: float | np.ndarray
    std_error: float | np.ndarray = float("nan")
    n: int = 0
    method: str = ""
    extras: dict = field(default_factory=dict)

    def confidence_interval(self, level=0.95):
        alpha2 = (1.0 - level) / 2.0
        df = self.extras.get("df")
        z = _t_ppf(1.0 - alpha2, df) if df else _norm_ppf(1.0 - alpha2)
        lo = self.estimate - z * self.std_error
        hi = self.estimate + z * self.std_error
        return lo, hi

    def __float__(self):
        return float(np.asarray(self.estimate).reshape(()))

    def __repr__(self):
        return f"EstimateResult(estimate={self.estimate!r}, std_error={self.std_error!r}, method={self.method!r})"


@dataclass
class DescribeResult:
    """Descriptive summary of a sample (or each column of a 2-D array)."""

    n: int
    missing: int
    mean: float | np.ndarray
    std: float | np.ndarray
    min: float | np.ndarray
    q1: float | np.ndarray
    median: float | np.ndarray
    q3: float | np.ndarray
    max: float | np.ndarray
    skewness: float | np.ndarray
    kurtosis: float | np.ndarray

    def as_dict(self):
        return {
            "n": self.n, "missing": self.missing, "mean": self.mean, "std": self.std,
            "min": self.min, "q1": self.q1, "median": self.median, "q3": self.q3,
            "max": self.max, "skewness": self.skewness, "kurtosis": self.kurtosis,
        }

    def __repr__(self):
        return (f"DescribeResult(n={self.n}, mean={np.asarray(self.mean).round(4)!r}, "
                f"std={np.asarray(self.std).round(4)!r})")


@dataclass
class RegressionResult:
    """Fitted linear/GLM/penalized regression model.

    ``coef_``/``std_errors_``/``tvalues_`` (or ``zvalues_`` for GLM)/``pvalues_`` are
    aligned arrays over the design columns (intercept first, if fit). ``family``/
    ``link`` are set for GLM fits; OLS-only fields (``r2_``, ``f_stat_``, ...) and
    GLM-only fields (``deviance_``, ``dispersion_``, ...) are ``None`` where not
    applicable. ``predict``/``predict_proba`` and ``conf_int``/``summary`` are bound
    methods set by the fitting function (not dataclass fields) via ``__post_init__``
    of the concrete estimator.
    """

    coef_: np.ndarray
    std_errors_: np.ndarray
    pvalues_: np.ndarray
    method: str
    fit_intercept: bool = True
    tvalues_: np.ndarray | None = None
    zvalues_: np.ndarray | None = None
    fitted_: np.ndarray | None = None
    resid_: np.ndarray | None = None
    df_model_: float | None = None
    df_resid_: float | None = None
    loglik_: float | None = None
    aic_: float | None = None
    bic_: float | None = None
    r2_: float | None = None
    adj_r2_: float | None = None
    f_stat_: float | None = None
    f_pvalue_: float | None = None
    cov_params_: np.ndarray | None = None
    deviance_: float | None = None
    null_deviance_: float | None = None
    pseudo_r2_: float | None = None
    dispersion_: float | None = None
    family: str | None = None
    link: str | None = None
    n_iter_: int | None = None
    extras: dict = field(default_factory=dict)

    def conf_int(self, level=0.95):
        alpha2 = (1.0 - level) / 2.0
        df = self.df_resid_ if self.df_resid_ else None
        z = _t_ppf(1.0 - alpha2, df) if df else _norm_ppf(1.0 - alpha2)
        lo = self.coef_ - z * self.std_errors_
        hi = self.coef_ + z * self.std_errors_
        return lo, hi

    def predict(self, X):
        return self.extras["_predict"](X)

    def predict_proba(self, X):
        if "_predict_proba" not in self.extras:
            raise NotImplementedError("predict_proba is only defined for binomial models")
        return self.extras["_predict_proba"](X)

    def summary(self):
        lines = [f"{self.method} (n={len(self.resid_) if self.resid_ is not None else '?'})"]
        for i, (c, se, p) in enumerate(zip(self.coef_, self.std_errors_, self.pvalues_)):
            name = "intercept" if (i == 0 and self.fit_intercept) else f"x{i - int(self.fit_intercept) + 1}"
            lines.append(f"  {name:>10s}: coef={c:.6g}  se={se:.6g}  p={p:.4g}")
        if self.r2_ is not None:
            lines.append(f"  R2={self.r2_:.4f}  adj-R2={self.adj_r2_:.4f}")
        if self.deviance_ is not None:
            lines.append(f"  deviance={self.deviance_:.4f}  aic={self.aic_:.4f}")
        return "\n".join(lines)

    def __repr__(self):
        return f"RegressionResult(method={self.method!r}, coef_={np.round(self.coef_, 4)!r})"


@dataclass
class PCAResult:
    """Principal component analysis fit."""

    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    mean_: np.ndarray
    scores_: np.ndarray

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, scores):
        scores = np.asarray(scores, dtype=float)
        return scores @ self.components_ + self.mean_


@dataclass
class FactorResult:
    """Factor analysis fit (maximum likelihood or principal-axis)."""

    loadings_: np.ndarray
    uniquenesses_: np.ndarray
    communalities_: np.ndarray
    loglik_: float | None
    method: str
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, X):
        """Regression (Thomson) factor scores."""
        X = np.asarray(X, dtype=float)
        Z = (X - self.mean_) / self.scale_
        L = self.loadings_
        Psi_inv = np.diag(1.0 / self.uniquenesses_)
        M = np.linalg.solve(L.T @ Psi_inv @ L + np.eye(L.shape[1]), L.T @ Psi_inv)
        return Z @ M.T


@dataclass
class CanonicalCorrelationResult:
    """Canonical correlation analysis fit between two variable sets."""

    correlations_: np.ndarray
    x_weights_: np.ndarray
    y_weights_: np.ndarray
    x_scores_: np.ndarray
    y_scores_: np.ndarray
    stats_: list


@dataclass
class DiscriminantResult:
    """Linear/quadratic discriminant analysis fit."""

    kind: str
    classes_: np.ndarray
    priors_: np.ndarray
    means_: np.ndarray
    coef_: np.ndarray | None
    intercept_: np.ndarray | None
    extras: dict = field(default_factory=dict)

    def predict(self, X):
        return self.extras["_predict"](X)

    def predict_proba(self, X):
        return self.extras["_predict_proba"](X)

    def transform(self, X):
        return self.extras["_transform"](X)


@dataclass
class ClusterResult:
    """Clustering fit (k-means or hierarchical agglomerative)."""

    labels_: np.ndarray
    method: str
    centers_: np.ndarray | None = None
    inertia_: float | None = None
    linkage_: np.ndarray | None = None
    silhouette_: float | None = None
    n_clusters: int = 0


@dataclass
class MDSResult:
    """Multidimensional scaling embedding."""

    embedding_: np.ndarray
    stress_: float
    method: str
    eigenvalues_: np.ndarray | None = None
    n_iter_: int | None = None
