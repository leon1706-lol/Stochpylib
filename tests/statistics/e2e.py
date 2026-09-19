"""End-to-end API sweep for stochpylib.statistics: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import math

import numpy as np
import pytest

from stochpylib import statistics as mod
from stochpylib.distributions import Gamma, Normal

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)
X1 = _RNG.normal(5.0, 2.0, 120)
X2 = _RNG.normal(6.0, 2.0, 110)
X3 = _RNG.normal(7.0, 2.0, 100)
XM = _RNG.normal(size=(200, 3))
Y_LIN = 1.0 + XM @ np.array([2.0, -1.0, 0.5]) + _RNG.normal(0, 0.5, 200)
Y_BIN = (_RNG.uniform(size=200) < 1 / (1 + np.exp(-(XM[:, 0] * 2.0)))).astype(int)
Y_CNT = _RNG.poisson(np.exp(0.3 + 0.5 * XM[:, 0]))
LABELS = (XM[:, 0] > 0).astype(int)


@exercise("mean")
def _mean():
    assert mod.mean(X1) == pytest.approx(np.mean(X1)) and mod.mean(X1, trim=0.1) != np.mean(X1)


@exercise("median")
def _median():
    assert mod.median(X1) == pytest.approx(np.median(X1))


@exercise("mode")
def _mode():
    val, count = mod.mode([1, 2, 2, 3, 3, 3])
    assert val == 3 and count == 3


@exercise("variance")
def _variance():
    assert mod.variance(X1) == pytest.approx(np.var(X1, ddof=1))


@exercise("std")
def _std():
    assert mod.std(X1) == pytest.approx(np.std(X1, ddof=1))


@exercise("quantile")
def _quantile():
    assert mod.quantile(X1, 0.3) == pytest.approx(np.quantile(X1, 0.3))


@exercise("iqr")
def _iqr():
    assert mod.iqr(X1) == pytest.approx(np.quantile(X1, 0.75) - np.quantile(X1, 0.25))


@exercise("skewness")
def _skewness():
    assert abs(mod.skewness(X1)) < 1.0


@exercise("kurtosis")
def _kurtosis():
    assert abs(mod.kurtosis(X1)) < 2.0


@exercise("covariance")
def _covariance():
    assert mod.covariance(X1[:100], X3) == pytest.approx(np.cov(X1[:100], X3, ddof=1)[0, 1])


@exercise("correlation")
def _correlation():
    r = mod.correlation(XM[:, 0], Y_LIN, method="pearson")
    assert 0.5 < float(r.estimate) < 1.0 and r.extras["pvalue"] < 1e-6


@exercise("describe")
def _describe():
    d = mod.describe(X1)
    assert d.n == 120 and d.min <= d.median <= d.max and "mean" in d.as_dict()


@exercise("DescribeResult")
def _describe_result():
    assert isinstance(mod.describe(X1), mod.DescribeResult)


@exercise("MLE")
def _mle():
    r = mod.MLE(Normal, X1)
    est = np.asarray(r.estimate, dtype=float)
    assert abs(est[0] - 5.0) < 0.6 and np.all(np.isfinite(np.asarray(r.std_error, dtype=float)))


@exercise("MOM")
def _mom():
    fitted = mod.MOM(Gamma, _RNG.gamma(2.0, 3.0, 5000))
    assert isinstance(fitted, Gamma) and abs(fitted.shape - 2.0) < 0.3


@exercise("bayesian_estimator")
def _bayes():
    r = mod.bayesian_estimator("normal", X1, prior=(0.0, 10.0), sigma=2.0)
    assert abs(float(r.estimate) - 5.0) < 0.6
    lo, hi = r.confidence_interval(0.95)
    assert lo < r.estimate < hi


@exercise("confidence_interval")
def _ci():
    lo, hi = mod.confidence_interval(X1, kind="mean")
    assert lo < np.mean(X1) < hi
    lo_p, hi_p = mod.confidence_interval((30, 100), kind="proportion", method="wilson")
    assert 0.2 < lo_p < 0.3 < hi_p < 0.4


@exercise("bootstrap_ci")
def _bootstrap():
    r = mod.bootstrap_ci(X1, lambda a: np.mean(a, axis=-1), n_boot=300, method="percentile",
                         random_state=1)
    lo, hi = r.extras["conf_int"]
    assert lo < np.mean(X1) < hi


@exercise("jackknife")
def _jackknife():
    r = mod.jackknife(X1, lambda a: np.mean(a))
    assert float(r.estimate) == pytest.approx(np.mean(X1))
    assert float(r.std_error) == pytest.approx(np.std(X1, ddof=1) / math.sqrt(120), rel=1e-6)


@exercise("delta_method")
def _delta():
    r = mod.delta_method(lambda t: math.log(t[0]), np.array([5.0]), np.array([[0.04]]))
    assert float(r.std_error) == pytest.approx(0.2 / 5.0, rel=1e-5)


@exercise("profile_likelihood")
def _profile():
    def loglik(params, data):
        mu, sigma = params
        return -1e18 if sigma <= 0 else float(np.sum(Normal(mu, sigma).pdf(data) * 0 + np.log(Normal(mu, sigma).pdf(data))))

    r = mod.profile_likelihood(loglik, np.array([np.mean(X1), np.std(X1)]), index=0, data=X1)
    lo, hi = r.extras["conf_int"]
    assert lo < np.mean(X1) < hi


@exercise("EstimateResult")
def _estimate_result():
    r = mod.EstimateResult(estimate=1.0, std_error=0.1, n=10)
    lo, hi = r.confidence_interval()
    assert lo < 1.0 < hi and float(r) == 1.0


@exercise("z_test")
def _z():
    r = mod.z_test(X1, mu0=5.0, sigma=2.0)
    assert 0 <= r.pvalue <= 1 and r.method


@exercise("t_test")
def _t():
    r = mod.t_test(X1, X2, equal_var=False)
    assert r.pvalue < 0.05 and r.reject() and r.estimate < 0


@exercise("chi2_test")
def _chi2():
    r = mod.chi2_test(np.array([[30, 10], [10, 30]]))
    assert r.pvalue < 1e-3


@exercise("f_test")
def _f():
    r = mod.f_test(X1, 3 * X2)
    assert r.pvalue < 1e-6


@exercise("ANOVA")
def _anova():
    r = mod.ANOVA(X1, X2, X3)
    assert r.pvalue < 1e-4 and r.table


@exercise("MANOVA")
def _manova():
    r = mod.MANOVA(XM[:100], LABELS[:100])
    assert 0 <= r.pvalue <= 1


@exercise("mann_whitney")
def _mann_whitney():
    assert mod.mann_whitney(X1, X3).pvalue < 1e-4


@exercise("wilcoxon")
def _wilcoxon():
    assert mod.wilcoxon(X1[:100], X3).pvalue < 1e-4


@exercise("ks_test")
def _ks():
    assert mod.ks_test(X1, "Normal", args=(5.0, 2.0)).pvalue > 0.01
    assert mod.ks_test(X1, X3).pvalue < 1e-3


@exercise("shapiro_wilk")
def _shapiro():
    assert mod.shapiro_wilk(X1).pvalue > 0.001


@exercise("levene")
def _levene():
    assert mod.levene(X1, 3 * X2).pvalue < 1e-6


@exercise("bartlett")
def _bartlett():
    assert mod.bartlett(X1, X2).pvalue > 0.001


@exercise("tukey_hsd")
def _tukey():
    r = mod.tukey_hsd(X1, X2, X3)
    assert len(r.table) == 3 and min(row["p"] for row in r.table) < 0.05


@exercise("bonferroni")
def _bonferroni():
    r = mod.bonferroni([0.001, 0.02, 0.3], method="holm")
    assert r.table[0]["adjusted_pvalue"] <= r.table[1]["adjusted_pvalue"]


@exercise("TestResult")
def _test_result():
    r = mod.TestResult(statistic=2.0, pvalue=0.01)
    assert r.reject(0.05) and "TestResult" in repr(r)


@exercise("linear_regression")
def _ols():
    r = mod.linear_regression(XM, Y_LIN, cov_type="HC3")
    assert np.allclose(r.coef_, [1.0, 2.0, -1.0, 0.5], atol=0.3) and r.r2_ > 0.9
    assert r.predict(XM[:3]).shape == (3,) and "coef" in r.summary()


@exercise("logistic_regression")
def _logistic():
    r = mod.logistic_regression(XM, Y_BIN)
    assert r.coef_[1] > 0.5 and np.all((r.predict_proba(XM) >= 0) & (r.predict_proba(XM) <= 1))


@exercise("poisson_regression")
def _poisson():
    r = mod.poisson_regression(XM, Y_CNT)
    assert abs(r.coef_[1] - 0.5) < 0.3


@exercise("ridge")
def _ridge():
    r = mod.ridge(XM, Y_LIN, alpha=1.0)
    assert np.allclose(r.coef_[1:], [2.0, -1.0, 0.5], atol=0.3)


@exercise("lasso")
def _lasso():
    r = mod.lasso(XM, Y_LIN, alpha=0.05)
    assert abs(r.coef_[1] - 2.0) < 0.3


@exercise("elastic_net")
def _enet():
    r = mod.elastic_net(XM, Y_LIN, alpha=0.05, l1_ratio=0.5)
    assert abs(r.coef_[1] - 2.0) < 0.3


@exercise("quantile_regression")
def _quantreg():
    r = mod.quantile_regression(XM, Y_LIN, q=0.5)
    assert abs(r.coef_[1] - 2.0) < 0.3


@exercise("glm")
def _glm():
    r = mod.glm(XM, Y_CNT, family="poisson", link="log")
    assert r.family == "poisson" and np.isfinite(r.deviance_)


@exercise("RegressionResult")
def _regression_result():
    r = mod.linear_regression(XM, Y_LIN)
    assert isinstance(r, mod.RegressionResult)
    lo, hi = r.conf_int()
    assert np.all(lo < r.coef_) and np.all(r.coef_ < hi)


@exercise("PCA")
def _pca():
    r = mod.PCA(XM, n_components=2)
    assert r.scores_.shape == (200, 2) and np.all(np.diff(r.explained_variance_) <= 0)
    assert np.allclose(r.inverse_transform(r.transform(XM[:5])).shape, (5, 3))


@exercise("PCAResult")
def _pca_result():
    assert isinstance(mod.PCA(XM), mod.PCAResult)


@exercise("factor_analysis")
def _fa():
    L = np.array([[0.9, 0.0], [0.8, 0.1], [0.1, 0.9], [0.0, 0.8]])
    F = _RNG.normal(size=(400, 2))
    Xf = F @ L.T + 0.3 * _RNG.normal(size=(400, 4))
    r = mod.factor_analysis(Xf, n_factors=2, method="pa", rotation="varimax")
    assert r.loadings_.shape == (4, 2) and r.transform(Xf[:5]).shape == (5, 2)


@exercise("FactorResult")
def _fa_result():
    F = _RNG.normal(size=(300, 2))
    Xf = np.column_stack([F[:, 0], F[:, 0] + 0.3 * _RNG.normal(size=300),
                          F[:, 1], F[:, 1] + 0.3 * _RNG.normal(size=300)])
    assert isinstance(mod.factor_analysis(Xf, n_factors=2, method="pa"), mod.FactorResult)


@exercise("canonical_correlation")
def _cca():
    Y = XM[:, :2] + 0.5 * _RNG.normal(size=(200, 2))
    r = mod.canonical_correlation(XM, Y)
    assert r.correlations_[0] > 0.7 and r.x_scores_.shape == (200, 2)


@exercise("CanonicalCorrelationResult")
def _cca_result():
    Y = XM[:, :2] + 0.5 * _RNG.normal(size=(200, 2))
    assert isinstance(mod.canonical_correlation(XM, Y), mod.CanonicalCorrelationResult)


@exercise("discriminant_analysis")
def _lda():
    r = mod.discriminant_analysis(XM, LABELS, kind="lda")
    assert np.mean(r.predict(XM) == LABELS) > 0.9
    assert r.predict_proba(XM[:4]).shape == (4, 2)


@exercise("DiscriminantResult")
def _lda_result():
    assert isinstance(mod.discriminant_analysis(XM, LABELS, kind="qda"), mod.DiscriminantResult)


@exercise("cluster_analysis")
def _cluster():
    Xc = np.vstack([XM[:100] + 8.0, XM[100:]])
    r = mod.cluster_analysis(Xc, n_clusters=2, random_state=0)
    assert r.labels_.shape == (200,) and len(set(r.labels_)) == 2 and r.centers_.shape == (2, 3)


@exercise("ClusterResult")
def _cluster_result():
    r = mod.cluster_analysis(XM, n_clusters=3, method="hierarchical")
    assert isinstance(r, mod.ClusterResult) and r.n_clusters == 3


@exercise("MDS")
def _mds():
    D = np.sqrt(((XM[:40, None, :] - XM[None, :40, :]) ** 2).sum(-1))
    r = mod.MDS(D, n_components=2)
    assert r.embedding_.shape == (40, 2) and np.isfinite(r.stress_)


@exercise("MDSResult")
def _mds_result():
    D = np.sqrt(((XM[:20, None, :] - XM[None, :20, :]) ** 2).sum(-1))
    assert isinstance(mod.MDS(D, n_components=2, method="smacof", random_state=0), mod.MDSResult)


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
