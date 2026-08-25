"""Tests for stochpylib.survival.

Validation strategy: known-DGP simulation ground truths (censored exponential/
Weibull/Cox/AFT/competing-risks generative models), closed-form identities
(KM↔NA consistency, Aalen-Johansen + KM = 1), hand-computable mini examples,
and — when lifelines is installed (dev-only extra) — direct cross-checks of
KM curves, Nelson-Aalen increments, Cox coefficients and log-rank statistics
against the reference implementation via ``pytest.importorskip``.

All randomness is seeded.
"""

import numpy as np
import pytest
from scipy import stats

from stochpylib.survival import (
    AalenAdditiveModel,
    AcceleratedFailureTime,
    BreslowEstimator,
    CauseSpecificHazard,
    CompetingRisksModel,
    CoxProportionalHazards,
    CumulativeHazard,
    EmpiricalSurvival,
    ExponentialSurvival,
    FineGrayModel,
    FlemingHarrington,
    GompertzSurvival,
    HazardFunction,
    KaplanMeier,
    LifeTable,
    LogLogisticSurvival,
    LogNormalSurvival,
    LogRankTest,
    MeanResidualLife,
    NelsonAalen,
    PetoTest,
    ResidualLifetime,
    StratifiedCox,
    SurvivalFunction,
    TaroneWareTest,
    WeibullSurvival,
    WilcoxonSurvival,
)
from stochpylib.survival.tests import _weighted_logrank

# ------------------------------------------------------------------ helpers


def _censored_exponential(rate, n, censor_lo=0.2, censor_hi=8.0, seed=0):
    rng = np.random.default_rng(seed)
    tt = rng.exponential(1 / rate, n)
    cc = rng.uniform(censor_lo, censor_hi, n)
    return np.minimum(tt, cc), (tt <= cc).astype(int)


def _two_arm(rate_a, rate_b, n, seed=3):
    ta, ea = _censored_exponential(rate_a, n, seed=seed)
    tb, eb = _censored_exponential(rate_b, n, seed=seed + 100)
    t = np.r_[ta, tb]
    e = np.r_[ea, eb]
    g = np.repeat(["A", "B"], n)
    x = np.r_[np.zeros(n), np.ones(n)]
    return t, e, g, x


# ---------------------------------------------------------------- KM / NA

def test_km_matches_closed_form_exponential():
    t, e = _censored_exponential(.5, 20000, seed=0)
    km = KaplanMeier().fit(t, e)
    for tt in (1.0, 2.0, 4.0):
        assert abs(float(km.predict([tt])[0]) - np.exp(-.5 * tt)) < .015


def test_km_median_survival_time():
    t, e = _censored_exponential(.5, 20000, seed=1)
    km = KaplanMeier().fit(t, e)
    assert abs(km.median_survival_time_ - np.log(2) / .5) < .05


def test_km_confidence_interval_covers_truth():
    t, e = _censored_exponential(.5, 20000, seed=2)
    km = KaplanMeier().fit(t, e)
    ci = km.confidence_interval_
    idx = int(np.searchsorted(ci["time"], 2.0))
    assert ci["lower"][idx] <= np.exp(-1.0) <= ci["upper"][idx]


def test_km_hand_computed_tiny_case():
    # events at t=1 (n=4), t=2 (n=3), censored at 3, event at t=4 (n=1)
    km = KaplanMeier().fit(np.array([1., 2., 3., 4.]),
                           np.array([1, 1, 0, 1]))
    assert np.allclose(km.survival_function_["time"], [1., 2., 4.])
    assert np.allclose(np.round(km.survival_function_["value"], 4),
                       [.75, .5, 0.])


def test_nelson_aalen_matches_minus_log_hazards():
    t, e = _censored_exponential(.5, 20000, seed=4)
    na = NelsonAalen().fit(t, e)
    for tt in (1.0, 2.0):
        assert abs(float(na.predict([tt])[0]) - .5 * tt) < .06


def test_na_and_km_consistent_small_ties():
    t = np.array([1., 2., 2., 3.])
    e = np.array([1, 0, 1, 1])
    km = KaplanMeier().fit(t, e)
    na = NelsonAalen().fit(t, e)
    # S(t) ≈ exp(-H(t)) up to tie correction; check loose consistency at end
    assert float(km.predict([3.])[0]) < float(np.exp(-float(
        na.predict([3.])[0])) + 0.02)


def test_life_table_fields_and_monotone_survival():
    t, e = _censored_exponential(.5, 2000, seed=6)
    lt = LifeTable().fit(t, e, width=1.0)
    assert len(lt.survival_) == len(lt.n_deaths_)
    assert np.all(np.diff(lt.survival_) <= 1e-12)


def test_empirical_survival_basic():
    es = EmpiricalSurvival().fit(np.array([1., 2., 3., 4.]))
    assert abs(float(es.predict([2.5])[0]) - .5) < 1e-12
    assert float(es.predict([10.])[0]) == 0.0


def test_breslow_estimator_positive_and_increasing():
    t, e, g, x = _two_arm(.5, .25, 1500, seed=7)
    be = BreslowEstimator().fit(t, e, np.exp(-x))
    assert be.values_[0] > 0
    assert np.all(np.diff(be.values_) >= -1e-12)


# ---------------------------------------------------------------- parametric

def test_weibull_parametric_recovery_under_censoring():
    rng = np.random.default_rng(11)
    tt = rng.weibull(1.5, 3000) * 10
    cc = rng.uniform(2, 30, 3000)
    t = np.minimum(tt, cc)
    e = (tt <= cc).astype(int)
    wb = WeibullSurvival().fit(t, e)
    assert abs(wb.params_["shape"] - 1.5) < .15
    assert abs(wb.params_["scale"] - 10) < 1.2
    assert wb.aic_ < ExponentialSurvival().fit(t, e).aic_


def test_exponential_closed_form_mle():
    rng = np.random.default_rng(12)
    t = rng.exponential(2.0, 5000)
    ex = ExponentialSurvival().fit(t, np.ones(5000, dtype=int))
    assert abs(ex.params_["rate"] - .5) < .02
    expected = float(np.exp(-2.0 * ex.params_["rate"]))
    assert abs(float(ex.survival_([2.0])[0]) - expected) < 1e-9


@pytest.mark.parametrize("cls", [LogNormalSurvival, LogLogisticSurvival,
                                 GompertzSurvival])
def test_other_parametrics_fit_finitely(cls):
    t, e = _censored_exponential(.5, 2500, seed=13)
    m = cls().fit(t, e)
    assert np.isfinite(m.aic_)
    assert np.all(np.isfinite(list(m.params_.values())))


# ---------------------------------------------------------------- functions

def test_survival_function_wrapper_from_data_and_distribution():
    from stochpylib.distributions import Weibull
    t, e = _censored_exponential(.5, 4000, seed=14)
    sf = SurvivalFunction(durations=t, events=e)
    assert abs(float(sf.predict([2.0])[0]) - np.exp(-1.0)) < .03
    sfd = SurvivalFunction(source=Weibull(1.5, 10.0))
    assert abs(float(sfd.predict([10.0])[0]) - np.exp(-1.0)) < 1e-9


def test_mrl_distribution_source_exact():
    from scipy.special import gamma
    from stochpylib.distributions import Weibull
    rl = ResidualLifetime(source=Weibull(1.5, 10.0))
    theory = 10.0 * gamma(1 + 1 / 1.5)
    assert abs(rl.value(0) - theory) < 0.05
    mrl = MeanResidualLife(source=Weibull(1.5, 10.0))
    assert mrl.curve([0.0]).size == 1


def test_cumulative_hazard_and_hazard_function_from_data():
    t, e = _censored_exponential(.5, 6000, seed=15)
    ch = CumulativeHazard(durations=t, events=e)
    assert abs(float(ch.predict([2.0])[0]) - 1.0) < .08
    hf = HazardFunction(durations=t, events=e)
    assert float(hf.predict([2.0])[0]) > 0


# ---------------------------------------------------------------- Cox

def test_cox_recovers_known_coefficient_sign_and_size():
    t, e, g, x = _two_arm(.5, .25, 3000, seed=16)
    cph = CoxProportionalHazards().fit(t, e, x)
    assert abs(cph.coefficients_[0] - np.log(.5)) < .1
    assert cph.concordance_index_ > .55
    se = cph.standard_errors_[0]
    assert 0 < se < 0.2
    assert np.isfinite(cph.p_values_[0])


def test_cox_efron_vs_breslow_agree_without_ties_and_work_with_them():
    t, e, g, x = _two_arm(.5, .25, 1500, seed=17)
    ef = CoxProportionalHazards(ties="efron").fit(t, e, x).coefficients_[0]
    br = CoxProportionalHazards(ties="breslow").fit(t, e, x).coefficients_[0]
    assert abs(ef - br) < .05          # no ties -> identical estimates

    td = np.round(np.minimum(t, 8) * 4) / 4   # forced heavy ties
    ef_t = CoxProportionalHazards(ties="efron").fit(td, e, x).coefficients_[0]
    br_t = CoxProportionalHazards(ties="breslow").fit(
        td, e, x).coefficients_[0]
    assert abs(ef_t - br_t) > 1e-6     # ties: methods genuinely differ
    assert ef_t * np.log(.5) > 0       # same direction as truth
    assert abs(ef_t) < 2.5             # both remain sane under heavy ties


def test_cox_baseline_breslow_estimator_consistency():
    t, e, g, x = _two_arm(.5, .25, 2000, seed=18)
    cph = CoxProportionalHazards().fit(t, e, x)
    H = np.asarray(cph.baseline_.predict(np.array([1.0, 2.0])), dtype=float)
    assert H[0] > 0 and H[1] > H[0]


def test_stratified_cox_runs_and_gives_finite_beta():
    t, e, g, x = _two_arm(.5, .25, 2000, seed=19)
    z = np.random.default_rng(20).exponential(1, len(t))
    sc = StratifiedCox().fit(t, e, z[:, None], g)
    assert np.isfinite(sc.coefficients_[0])
    assert set(sc.baseline_by_stratum_) == {"A", "B"}


# ---------------------------------------------------------------- AFT / Aalen

def test_aft_recovers_negative_effect():
    t, e, g, x = _two_arm(.5, .25, 2500, seed=21)
    aft = AcceleratedFailureTime().fit(t, e, x)
    assert aft.coefficients_[0] < 0            # x=1 lives longer
    assert aft.shape_ > 0.5
    med = aft.predict_median(np.array([[0.0], [1.0]]))
    assert med[1] > med[0]


def test_aalen_additive_directional_hazards():
    t, e, g, x = _two_arm(.5, .25, 2000, seed=22)
    am = AalenAdditiveModel().fit(t, e, np.column_stack(
        [np.ones(len(t)), x]))
    h0 = float(am.predict([1.0, 0.0], [2.0])[0])
    h1 = float(am.predict([1.0, 1.0], [2.0])[0])
    assert h0 > h1                             # arm A has higher hazard
    assert h0 > 0.5                            # ~ theoretical H_A(2)=1


# ---------------------------------------------------------------- log-rank

def test_logrank_family_separates_and_calibrates():
    t, e, g, _x = _two_arm(.5, .25, 1200, seed=23)
    for cls, kw in [(LogRankTest, {}), (WilcoxonSurvival, {}),
                    (TaroneWareTest, {}), (PetoTest, {}),
                    (FlemingHarrington, {"rho": 1.0, "gamma": 0.5})]:
        res = cls(**kw).fit(t, e, g)
        assert res.p_value_ < 1e-10, cls.__name__

    # calibration: identical arms -> uniform p across seeds
    ps = []
    for seed in range(12):
        rng = np.random.default_rng(500 + seed)
        n = 250
        ta = np.minimum(rng.exponential(2, n), rng.uniform(.2, 8, n))
        ea = (rng.exponential(2, n) <= rng.uniform(.2, 8, n)).astype(int)
        tb = np.minimum(rng.exponential(2, n), rng.uniform(.2, 8, n))
        eb = (rng.exponential(2, n) <= rng.uniform(.2, 8, n)).astype(int)
        r = LogRankTest().fit(np.r_[ta, tb], np.r_[ea, eb],
                              np.repeat(["A", "B"], n))
        ps.append(r.p_value_)
    frac_sig = np.mean(np.array(ps) < .05)
    assert frac_sig <= 2 / 12          # nominal 5% level, small-sample slack


def test_weighted_variants_span_reasonable_range_on_separated_data():
    t, e, g, _x = _two_arm(.5, .25, 1200, seed=24)
    stats_lr = []
    for cls, kw in [(LogRankTest, {}), (WilcoxonSurvival, {}),
                    (TaroneWareTest, {}), (PetoTest, {}),
                    (FlemingHarrington, {"rho": 1.0, "gamma": 0.0})]:
        r = cls(**kw).fit(t, e, g)
        stats_lr.append(r.test_statistic_)
    assert all(s > 10 for s in stats_lr)


def test_weighted_logrank_multigroup_three_arms():
    rng = np.random.default_rng(25)
    parts = []
    labs = []
    for gi, rate in enumerate((.3, .6, 1.2)):
        tt = np.minimum(rng.exponential(1 / rate, 700),
                        rng.uniform(.2, 8, 700))
        ee = (rng.exponential(1 / rate, 700) <=
              rng.uniform(.2, 8, 700)).astype(int)
        parts.append(tt)
        parts.append(ee)
        labs.append(np.full(700, f"G{gi}"))
    tt = parts[0::2]
    ee = parts[1::2]
    t = np.concatenate(tt)
    e = np.concatenate(ee)
    g = np.concatenate(labs)
    res = _weighted_logrank(t, e, g, lambda u, n, N, s: 1.0)
    assert res["degrees_of_freedom"] == 2
    assert res["p_value"] < 1e-20


# ---------------------------------------------------------------- competing risks

def _cr_dgp(n=2500, beta=-0.8, seed=99):
    rng = np.random.default_rng(seed)
    x = rng.choice([0., 1.], n)
    h1 = .3 * np.exp(beta * x)
    c1 = -np.log(rng.random(n)) / h1
    c2 = -np.log(rng.random(n)) / .4
    cc = rng.uniform(.1, 10, n)
    T = np.minimum.reduce([c1, c2, cc])
    cause = np.where((c1 <= c2) & (c1 <= cc), 1,
                     np.where(c2 < cc, 2, 0))
    return T, cause, x


def test_competing_risks_identity_exact():
    T, C, _x = _cr_dgp()
    crm = CompetingRisksModel().fit(T, C)
    assert crm.check_identity() < 1e-9


def test_cif_values_within_bounds_and_ordered():
    T, C, _x = _cr_dgp()
    crm = CompetingRisksModel().fit(T, C)
    for k in crm.causes_:
        vals = crm.cif_[k].cif_["value"]
        assert np.all((vals >= 0) & (vals <= 1))
        assert np.all(np.diff(vals) >= -1e-12)


def test_cause_specific_hazard_predicts():
    T, C, _x = _cr_dgp()
    csh = CauseSpecificHazard().fit(T, C, cause_of_interest=2)
    h1 = float(csh.predict([1.0])[0])
    h3 = float(csh.predict([3.0])[0])
    assert h3 > h1 >= 0


# ---------------------------------------------------------------- FineGray

def test_finegray_detects_subdistribution_effect():
    T, C, x = _cr_dgp(beta=-0.8, seed=26)
    fg = FineGrayModel().fit(T, C, np.column_stack([x]),
                             cause_of_interest=1)
    assert fg.coefficients_[0] < -0.2      # correct direction, attenuated
    assert fg.z_scores_[0] < -3            # strongly significant
    assert fg.standard_errors_[0] > 0


def test_finegray_null_effect_not_significant():
    T, C, x = _cr_dgp(beta=0.0, seed=27)
    fg = FineGrayModel().fit(T, C, np.column_stack([x]),
                             cause_of_interest=1)
    assert abs(fg.z_scores_[0]) < 3


# ---------------------------------------------------------------- methods & wiring

def test_top_level_wiring_and_conformance_counts():
    import stochpylib
    assert stochpylib.__all__[-1] == "timeseries" or "survival" \
        in stochpylib.__all__
    assert hasattr(stochpylib, "survival")
    from stochpylib import survival
    expected = {"KaplanMeier", "NelsonAalen", "LifeTable", "EmpiricalSurvival",
                "BreslowEstimator", "SurvivalFunction", "HazardFunction",
                "CumulativeHazard", "ResidualLifetime", "MeanResidualLife",
                "WeibullSurvival", "ExponentialSurvival", "LogNormalSurvival",
                "LogLogisticSurvival", "GompertzSurvival",
                "CoxProportionalHazards", "AcceleratedFailureTime",
                "AalenAdditiveModel", "FineGrayModel", "StratifiedCox",
                "LogRankTest", "WilcoxonSurvival", "TaroneWareTest",
                "PetoTest", "FlemingHarrington", "CumulativeIncidenceFunction",
                "CompetingRisksModel", "CauseSpecificHazard"}
    missing = expected - set(survival.__all__)
    assert not missing, missing


def test_doctests_pass():
    import doctest
    import stochpylib.survival._base as _base
    import stochpylib.survival.nonparametric as _np_mod
    assert doctest.testmod(_base).failed == 0
    assert doctest.testmod(_np_mod).failed == 0


# ---------------------------------------------------------------- lifelines oracle
# dev-only reference implementation; these cross-checks run when the optional
# `lifelines` extra is installed (CI installs [dev]) and skip cleanly otherwise.


def test_km_curve_matches_lifelines():
    lf = pytest.importorskip("lifelines")
    from lifelines import KaplanMeierFitter
    rng = np.random.default_rng(40)
    t = rng.exponential(2, 800)
    e = rng.integers(0, 2, 800).astype(int)
    ours = KaplanMeier().fit(t, e)
    lff = KaplanMeierFitter().fit(t, e)
    grid = np.linspace(.1, 7, 40)
    a = ours.predict(grid)
    b = lff.predict(grid).values
    assert np.max(np.abs(a - b)) < 1e-10


def test_nelson_aalen_matches_lifelines():
    from lifelines import NelsonAalenFitter
    rng = np.random.default_rng(41)
    t = rng.exponential(2, 800)
    e = rng.integers(0, 2, 800).astype(int)
    ours = NelsonAalen().fit(t, e)
    lff = NelsonAalenFitter().fit(t, e)
    grid = np.linspace(.1, 7, 40)
    a = ours.predict(grid)
    b = lff.predict(grid).values
    assert np.max(np.abs(a - b)) < 1e-9


def test_cox_coefficients_match_lifelines():
    from lifelines import CoxPHFitter
    rng = np.random.default_rng(42)
    n = 600
    x = rng.choice([0., 1.], n)
    lam = .5 * np.exp(np.log(.5) * x)
    t = rng.exponential(1 / lam)
    c = rng.uniform(.2, 8, n)
    dur = np.minimum(t, c)
    ev = (t <= c).astype(int)
    df = np.column_stack([dur, ev, x])
    ours = CoxProportionalHazards(ties="efron").fit(dur, ev, x[:, None])
    lff = CoxPHFitter()
    import pandas as pd
    lff.fit(pd.DataFrame(df, columns=["T", "E", "x"]), "T", "E")
    assert abs(ours.coefficients_[0] -
               lff.params_["x"]) < max(1e-6, 0.05 * abs(lff.params_["x"]))


# ---------------------------------------------------------------- V0.4.1 audit additions

def test_km_all_censored_median_is_inf():
    km = KaplanMeier().fit(np.array([1., 2., 3.]), np.zeros(3, dtype=int))
    assert km.median_survival_time_ == float("inf")
    assert float(km.predict([5.0])[0]) == 1.0


def test_na_all_censored_h_is_zero():
    na = NelsonAalen().fit(np.array([1., 2.]), np.zeros(2, dtype=int))
    assert float(na.predict([5.0])[0]) == 0.0


def test_step_evaluate_cumulative_hazard_default_is_zero():
    from stochpylib.survival._base import SurvivalFitter
    arr = np.array([(1.0, 0.5)], dtype=[("time", float), ("value", float)])
    result = SurvivalFitter._step_evaluate(arr, [0.0, 2.0], default=0.0)
    assert result[0] == 0.0
    assert result[1] == 0.5


def test_gompertz_small_b_approximates_exponential():
    g = GompertzSurvival()
    g.params_ = {"a": 0.5, "b": 1e-15}
    sv = g._survival(np.array([1.0, 2.0]), g._theta())
    assert np.allclose(sv, np.exp(-0.5 * np.array([1., 2.])))


def test_loglogistic_density_positive_at_alpha():
    ll = LogLogisticSurvival()
    ll.params_ = {"alpha": 5.0, "beta": 2.0}
    d = float(ll._density(np.array([5.0]), ll._theta())[0])
    assert d > 0


def test_stratified_cox_three_strata():
    rng = np.random.default_rng(30)
    t3 = np.r_[rng.exponential(2, 500), rng.exponential(3, 500),
               rng.exponential(1.5, 500)]
    c3 = rng.uniform(.3, 10, 1500)
    d3 = np.minimum(t3, c3)
    e3 = (t3 <= c3).astype(int)
    z3 = rng.standard_normal(1500)
    strata3 = np.repeat(["X", "Y", "Z"], 500)
    sc = StratifiedCox().fit(d3, e3, z3[:, None], strata3)
    assert len(sc.baseline_by_stratum_) == 3
    assert np.isfinite(sc.coefficients_[0])


def test_logrank_three_groups_df_two():
    rng = np.random.default_rng(31)
    parts_t = []
    parts_e = []
    labs = []
    for gi, rate in enumerate((.3, .6, 1.2)):
        tt = np.minimum(rng.exponential(1 / rate, 500),
                        rng.uniform(.2, 8, 500))
        ee = (rng.exponential(1 / rate, 500) <=
              rng.uniform(.2, 8, 500)).astype(int)
        parts_t.append(tt)
        parts_e.append(ee)
        labs.append(np.full(500, f"G{gi}"))
    t = np.concatenate(parts_t)
    e = np.concatenate(parts_e)
    g = np.concatenate(labs)
    lr = LogRankTest().fit(t, e, g)
    assert lr.degrees_of_freedom_ == 2
    assert np.isfinite(lr.p_value_)


def test_competing_risks_three_causes_identity():
    rng = np.random.default_rng(32)
    cause3 = rng.choice([1, 2, 3], 800)
    tt3 = rng.exponential(2, 800)
    cc3 = rng.uniform(.2, 8, 800)
    T3 = np.minimum(tt3, cc3)
    C3 = np.where(tt3 <= cc3, cause3, 0)
    crm = CompetingRisksModel().fit(T3, C3)
    assert crm.check_identity() < 1e-9


def test_mrl_exponential_memoryless():
    from stochpylib.distributions import Exponential as DExp
    rl = ResidualLifetime(source=DExp(0.5))
    assert abs(rl.value(0) - 2.0) < .05
    assert abs(rl.value(3) - 2.0) < .05   # memoryless


def test_cumulative_hazard_from_parametric_integration():
    ws = WeibullSurvival()
    ws.params_ = {"shape": 1.0, "scale": 2.0}
    ch = CumulativeHazard(source=ws)
    assert abs(float(ch.predict([2.0])[0]) - 1.0) < .05


def test_hazard_function_from_distribution():
    from stochpylib.distributions import Exponential as DExp
    hf = HazardFunction(source=DExp(0.5))
    assert abs(float(hf.predict([3.0])[0]) - .5) < .01


def test_survival_function_from_parametric_model():
    ws = WeibullSurvival()
    ws.params_ = {"shape": 1.5, "scale": 10.0}
    sf = SurvivalFunction(source=ws)
    expected = np.exp(-(10 / 10) ** 1.5)
    assert abs(float(sf.predict([10.0])[0]) - expected) < 1e-8


def test_breslow_baseline_survival_in_unit_interval():
    rng = np.random.default_rng(33)
    t = np.minimum(rng.exponential(2, 500), rng.uniform(.2, 8, 500))
    be = BreslowEstimator().fit(t, np.ones(500, dtype=int), np.ones(500))
    bs = be.baseline_survival_(np.array([1.0, 2.0, 4.0]))
    assert np.all((bs >= 0) & (bs <= 1))


def test_cox_predict_partial_hazard_shape():
    rng = np.random.default_rng(34)
    x1 = rng.standard_normal(200)
    t_cox = rng.exponential(2, 200)
    c_cox = rng.uniform(.2, 8, 200)
    dur = np.minimum(t_cox, c_cox)
    ev = (t_cox <= c_cox).astype(int)
    m = CoxProportionalHazards().fit(dur, ev, x1[:, None])
    ph = m.predict_partial_hazard(x1[:10][:, None])
    assert ph.shape == (10,)


def test_aalen_multi_time_monotone():
    rng = np.random.default_rng(35)
    am = AalenAdditiveModel().fit(
        np.minimum(rng.exponential(2, 500), rng.uniform(.5, 8, 500)),
        np.ones(500, dtype=int), np.ones((500, 1)))
    times = np.array([0.5, 1.0, 2.0, 4.0])
    preds = am.predict([1.0], times)
    assert np.all(np.diff(preds) >= -1e-12)


def test_finegray_no_competing_risks_still_runs():
    rng = np.random.default_rng(36)
    T1 = rng.exponential(2, 300)
    C1 = rng.uniform(.1, 10, 300)
    cause1_only = np.where(T1 <= C1, 1, 0)
    fg = FineGrayModel().fit(T1, cause1_only,
                              np.column_stack([rng.random(300)]),
                             cause_of_interest=1)
    assert np.isfinite(fg.coefficients_[0])


def test_cox_perfect_separation_converges():
    rng = np.random.default_rng(37)
    t_sep = np.r_[rng.exponential(1, 50), rng.exponential(10, 50)]
    c_sep = rng.uniform(.5, 20, 100)
    d_sep = np.minimum(t_sep, c_sep)
    ev_sep = (t_sep <= c_sep).astype(int)
    x_sep = np.r_[np.zeros(50), np.ones(50)]
    m = CoxProportionalHazards().fit(d_sep, ev_sep, x_sep[:, None])
    assert np.isfinite(m.coefficients_[0])
