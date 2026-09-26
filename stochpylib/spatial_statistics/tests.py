"""Spatial autocorrelation tests: Moran's I, Geary's C, a unified front end (global or
local, any of the three statistics), and a nearest-neighbour distance test (Clark-Evans /
G-function). All four are plain functions (per the spec's ``()`` convention) returning a
``statistics.TestResult``.
"""

import numpy as np

from stochpylib.spatial_statistics._common import _rng
from stochpylib.spatial_statistics.point_processes import RipleyK
from stochpylib.spatial_statistics.weights import SpatialWeights
from stochpylib.statistics._common import _norm_cdf
from stochpylib.statistics._result import TestResult

__all__ = ["MoransI", "GearyC", "SpatialAutocorrelation", "NNDistanceTest"]


def _weights_matrix(weights):
    return weights.W if isinstance(weights, SpatialWeights) else np.asarray(weights, dtype=float)


def _cliff_ord_moments(W):
    """S0, S1, S2 (Cliff & Ord 1981) from a raw weights matrix."""
    n = W.shape[0]
    S0 = float(W.sum())
    S1 = float(0.5 * np.sum((W + W.T) ** 2))
    row, col = W.sum(axis=1), W.sum(axis=0)
    S2 = float(np.sum((row + col) ** 2))
    return n, S0, S1, S2


def _permutation_pvalue(stat_fn, z, W, permutations, random_state, alternative):
    rng = _rng(random_state)
    n = len(z)
    sims = np.empty(permutations)
    for i in range(permutations):
        perm = rng.permutation(n)
        sims[i] = stat_fn(z[perm], W)
    obs = stat_fn(z, W)
    if alternative == "greater":
        return float((1 + np.sum(sims >= obs)) / (1 + permutations))
    if alternative == "less":
        return float((1 + np.sum(sims <= obs)) / (1 + permutations))
    dev = np.abs(sims - sims.mean())
    obs_dev = abs(obs - sims.mean())
    return float((1 + np.sum(dev >= obs_dev)) / (1 + permutations))


def _moran_stat(z, W):
    n = len(z)
    S0 = W.sum()
    num = float(z @ (W @ z))
    den = float(z @ z)
    return (n / S0) * (num / den) if S0 > 0 and den > 0 else 0.0


def MoransI(values, weights, permutations=0, alternative="two-sided", random_state=None):
    """Moran's I spatial autocorrelation statistic (global)."""
    z = np.asarray(values, dtype=float).ravel()
    W = _weights_matrix(weights)
    n, S0, S1, S2 = _cliff_ord_moments(W)
    zc = z - z.mean()
    I = _moran_stat(zc, W)
    expected = -1.0 / (n - 1)
    b2 = (np.sum(zc ** 4) / n) / (np.sum(zc ** 2) / n) ** 2  # kurtosis
    S0sq = S0 ** 2
    var_norm = (n ** 2 * S1 - n * S2 + 3.0 * S0sq) / (S0sq * (n ** 2 - 1)) - expected ** 2
    var_rand = ((n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3.0 * S0sq)
                 - b2 * (n * (n - 1) * S1 - 2.0 * n * S2 + 6.0 * S0sq))
                / ((n - 1) * (n - 2) * (n - 3) * S0sq)) - expected ** 2
    var_rand = max(var_rand, 0.0)
    z_norm = (I - expected) / np.sqrt(var_norm) if var_norm > 0 else 0.0
    z_rand = (I - expected) / np.sqrt(var_rand) if var_rand > 0 else 0.0
    p_norm = float(2.0 * _norm_cdf(-abs(z_norm)))
    p_rand = float(2.0 * _norm_cdf(-abs(z_rand)))
    extras = {"expected": expected, "var_normality": var_norm, "var_randomization": var_rand,
              "z_norm": float(z_norm), "z_rand": float(z_rand), "p_norm": p_norm, "p_rand": p_rand}
    pvalue = p_rand
    if permutations > 0:
        p_sim = _permutation_pvalue(_moran_stat, zc, W, int(permutations), random_state, alternative)
        extras["p_sim"] = p_sim
        pvalue = p_sim
    elif alternative != "two-sided":
        pvalue = float(_norm_cdf(z_rand) if alternative == "less" else _norm_cdf(-z_rand))
    return TestResult(float(I), pvalue, n - 2, "no spatial autocorrelation", "Moran's I",
                       alternative, extras=extras)


def _geary_stat(z, W):
    n = len(z)
    S0 = W.sum()
    diff2 = (z[:, None] - z[None, :]) ** 2
    num = float(np.sum(W * diff2))
    den = float(np.sum(z ** 2))
    return ((n - 1) * num) / (2.0 * S0 * den) if S0 > 0 and den > 0 else 1.0


def GearyC(values, weights, permutations=0, alternative="two-sided", random_state=None):
    """Geary's C spatial autocorrelation statistic (global; E[C] = 1 under no autocorrelation)."""
    z = np.asarray(values, dtype=float).ravel()
    W = _weights_matrix(weights)
    n, S0, S1, S2 = _cliff_ord_moments(W)
    zc = z - z.mean()
    C = _geary_stat(zc, W)
    expected = 1.0
    b2 = (np.sum(zc ** 4) / n) / (np.sum(zc ** 2) / n) ** 2
    S0sq = S0 ** 2
    var_norm = ((2.0 * S1 + S2) * (n - 1) - 4.0 * S0sq) / (2.0 * (n + 1) * S0sq)
    var_rand = (((n - 1) * S1 * (n ** 2 - 3 * n + 3 - (n - 1) * b2)
                 - 0.25 * (n - 1) * S2 * (n ** 2 + 3 * n - 6 - (n ** 2 - n + 2) * b2)
                 + S0sq * (n ** 2 - 3 - (n - 1) ** 2 * b2))
                / (n * (n - 2) * (n - 3) * S0sq))
    var_rand = max(var_rand, 0.0)
    z_norm = (C - expected) / np.sqrt(var_norm) if var_norm > 0 else 0.0
    z_rand = (C - expected) / np.sqrt(var_rand) if var_rand > 0 else 0.0
    p_norm = float(2.0 * _norm_cdf(-abs(z_norm)))
    p_rand = float(2.0 * _norm_cdf(-abs(z_rand)))
    extras = {"expected": expected, "var_normality": var_norm, "var_randomization": var_rand,
              "z_norm": float(z_norm), "z_rand": float(z_rand), "p_norm": p_norm, "p_rand": p_rand}
    pvalue = p_rand
    if permutations > 0:
        p_sim = _permutation_pvalue(_geary_stat, zc, W, int(permutations), random_state, alternative)
        extras["p_sim"] = p_sim
        pvalue = p_sim
    elif alternative != "two-sided":
        pvalue = float(_norm_cdf(-z_rand) if alternative == "less" else _norm_cdf(z_rand))
    return TestResult(float(C), pvalue, n - 2, "no spatial autocorrelation", "Geary's C",
                       alternative, extras=extras)


def _getis_ord_general_g(z, W):
    num = float(np.sum(W * (z[:, None] * z[None, :])))
    off = ~np.eye(len(z), dtype=bool)
    den = float(np.sum(z[:, None] * z[None, :] * off))
    return num / den if den != 0 else 0.0


def SpatialAutocorrelation(values, weights, statistic="moran", local=False, permutations=0,
                            random_state=None, alternative="two-sided"):
    """Unified front end: global or local Moran/Geary/Getis-Ord spatial autocorrelation.

    ``statistic="getis_ord"`` (General G / Gi*) is classically defined for non-negative
    attribute values; it is still computed on mixed-sign data, but its usual
    interpretation (and the analytic ``expected`` moment) assumes ``values >= 0``.
    """
    z = np.asarray(values, dtype=float).ravel()
    W = _weights_matrix(weights)
    n = len(z)
    if not local:
        if statistic == "moran":
            return MoransI(z, W, permutations, alternative, random_state)
        if statistic == "geary":
            return GearyC(z, W, permutations, alternative, random_state)
        if statistic == "getis_ord":
            g_raw = _getis_ord_general_g(z, W)  # Getis-Ord General G uses raw levels, not centered
            EG = float(W.sum()) / (n * (n - 1))
            extras = {"expected": EG}
            pvalue = None
            if permutations > 0:
                rng = _rng(random_state)
                sims = np.array([_getis_ord_general_g(z[rng.permutation(n)], W)
                                  for _ in range(int(permutations))])
                extras["p_sim"] = pvalue = float((1 + np.sum(np.abs(sims - EG) >= abs(g_raw - EG)))
                                                  / (1 + permutations))
            return TestResult(float(g_raw), pvalue, None, "no spatial association",
                               "Getis-Ord General G", alternative, extras=extras)
        raise ValueError("statistic must be 'moran', 'geary' or 'getis_ord'")

    zc = z - z.mean()
    m2 = float(np.sum(zc ** 2)) / n
    if statistic == "moran":
        Ii = (zc / m2) * (W @ zc)
        global_res = MoransI(z, W)
        wi2 = np.sum(W ** 2, axis=1)
        b2 = (np.sum(zc ** 4) / n) / m2 ** 2
        EIi = -wi2 / (n - 1)
        name = "Local Moran's I"
    elif statistic == "geary":
        diff2 = (zc[:, None] - zc[None, :]) ** 2
        Ii = np.sum(W * diff2, axis=1) / m2
        EIi = np.sum(W, axis=1)  # under randomization, E[c_i] ~ sum_j w_ij (up to (n-1) scaling)
        name = "Local Geary's C"
    elif statistic == "getis_ord":
        row_sum = np.sum(W, axis=1)
        num = W @ z
        Ii = num / row_sum if np.all(row_sum > 0) else num
        EIi = np.full(n, z.mean())
        name = "Local Getis-Ord Gi*"
    else:
        raise ValueError("statistic must be 'moran', 'geary' or 'getis_ord'")

    extras = {"expected": EIi}
    pvalue = None
    if permutations > 0:
        rng = _rng(random_state)
        sims = np.empty((int(permutations), n))
        for s in range(int(permutations)):
            perm_z = zc[rng.permutation(n)] if statistic != "getis_ord" else z[rng.permutation(n)]
            if statistic == "moran":
                sims[s] = (perm_z / m2) * (W @ perm_z)
            elif statistic == "geary":
                diff2 = (perm_z[:, None] - perm_z[None, :]) ** 2
                sims[s] = np.sum(W * diff2, axis=1) / m2
            else:
                row_sum = np.sum(W, axis=1)
                sims[s] = (W @ perm_z) / np.where(row_sum > 0, row_sum, 1.0)
        dev = np.abs(sims - Ii[None, :])
        obs_dev = np.zeros(n)
        pvalue = (1.0 + np.sum(dev >= obs_dev[None, :], axis=0)) / (1.0 + permutations)
        extras["p_sim"] = pvalue
    return TestResult(Ii, pvalue, None, "no local spatial association", name, alternative,
                       extras=extras)


def NNDistanceTest(points, window, method="clark_evans", edge="donnelly", n_simulations=0,
                    random_state=None):
    """Nearest-neighbour distance test: Clark-Evans R, or the G-function CSR envelope test."""
    from stochpylib.spatial_statistics._common import _box_window, _pairwise

    X = np.asarray(points, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    W, vol = _box_window(window)
    n = len(X)
    lam = n / vol

    if method == "clark_evans":
        d_full = _pairwise(X)
        np.fill_diagonal(d_full, np.inf)
        nn = d_full.min(axis=1)
        r_obs = float(nn.mean())
        r_expected = 0.5 / np.sqrt(lam)
        r_null = r_expected  # the mean the z-test is measured against
        if edge == "donnelly" and W.shape[0] == 2:
            # Donnelly (1978): observed NN distances are edge-truncated (a point's true
            # nearest neighbour can lie outside the window), which biases both the mean
            # and variance of r_obs upward under CSR -- correct both for the z-test, but
            # leave R = r_obs / r_expected on the simple, universally-recognized scale.
            A = float(vol)
            perim = 2.0 * ((W[0, 1] - W[0, 0]) + (W[1, 1] - W[1, 0]))
            r_null = 0.5 * np.sqrt(A / n) + (0.0514 + 0.041 / np.sqrt(n)) * perim / n
            se = np.sqrt(0.070 * A / n ** 2 + 0.037 * perim * np.sqrt(A) / n ** 2.5)
        else:
            se = np.sqrt((4.0 - np.pi) / (4.0 * np.pi * lam * n))
        R = r_obs / r_expected
        z = (r_obs - r_null) / se if se > 0 else 0.0
        pvalue = float(2.0 * _norm_cdf(-abs(z)))
        extras = {"r_observed": r_obs, "r_expected": r_expected, "z": float(z), "se": float(se)}
        return TestResult(float(R), pvalue, None, "complete spatial randomness (CSR)",
                           "Clark-Evans nearest-neighbour test", "two-sided", extras=extras)

    if method == "G":
        d_full = _pairwise(X)
        np.fill_diagonal(d_full, np.inf)
        nn = np.sort(d_full.min(axis=1))
        r = np.linspace(0.0, float(nn[-1]) if n else 1.0, 30)
        g_obs = np.searchsorted(nn, r, side="right") / n
        g_theo = 1.0 - np.exp(-lam * np.pi * r ** 2) if W.shape[0] == 2 else 1.0 - np.exp(-lam * r)
        pvalue = None
        if n_simulations > 0:
            rng = _rng(random_state)
            devs = np.empty(int(n_simulations))
            for s in range(int(n_simulations)):
                pts = rng.uniform(W[:, 0], W[:, 1], size=(n, W.shape[0]))
                dp = _pairwise(pts)
                np.fill_diagonal(dp, np.inf)
                nnp = np.sort(dp.min(axis=1))
                g_s = np.searchsorted(nnp, r, side="right") / n
                devs[s] = np.max(np.abs(g_s - g_theo))
            obs_dev = float(np.max(np.abs(g_obs - g_theo)))
            pvalue = float((1 + np.sum(devs >= obs_dev)) / (1 + n_simulations))
        stat = float(np.max(np.abs(g_obs - g_theo)))
        extras = {"r": r, "g_observed": g_obs, "g_theoretical": g_theo}
        return TestResult(stat, pvalue, None, "complete spatial randomness (CSR)",
                           "G-function nearest-neighbour test", "two-sided", extras=extras)

    raise ValueError("method must be 'clark_evans' or 'G'")
