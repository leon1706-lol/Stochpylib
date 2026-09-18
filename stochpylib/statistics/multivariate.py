"""Multivariate methods: dimensionality reduction, factor analysis, canonical
correlation, discriminant analysis, clustering, and multidimensional scaling.
"""

import math

import numpy as np
from scipy import optimize, special

from stochpylib.statistics._common import _as_2d, _f_sf, _rng
from stochpylib.statistics._result import (
    CanonicalCorrelationResult, ClusterResult, DiscriminantResult, FactorResult, MDSResult, PCAResult,
)

__all__ = [
    "MDS", "PCA", "canonical_correlation", "cluster_analysis", "discriminant_analysis",
    "factor_analysis",
]


def PCA(X, n_components=None, standardize=False):
    """Principal component analysis via SVD of the (optionally standardized)
    centered data. Component signs are fixed so each component's largest-magnitude
    loading is positive (a deterministic convention; SVD signs are otherwise
    arbitrary)."""
    X = _as_2d(X, "X")
    n, p = X.shape
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1) if standardize else np.ones(p)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale

    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    for j in range(Vt.shape[0]):
        if Vt[j, np.argmax(np.abs(Vt[j]))] < 0:
            Vt[j] *= -1
            U[:, j] *= -1

    k = n_components or min(n, p)
    explained_var = s ** 2 / (n - 1)
    total_var = np.sum(explained_var)

    res = PCAResult(
        components_=Vt[:k], explained_variance_=explained_var[:k],
        explained_variance_ratio_=explained_var[:k] / total_var,
        singular_values_=s[:k], mean_=mean, scores_=(U[:, :k] * s[:k]),
    )
    res._scale = scale
    return res


def factor_analysis(X, n_factors, method="ml", rotation=None, standardize=True, max_iter=200):
    """Factor analysis on the correlation (``standardize=True``, the default and
    the ``statsmodels``-matching convention) or covariance matrix.

    ``method="ml"``: maximum-likelihood via profiled Jöreskog concentration (BFGS
    over the uniquenesses); ``method="pa"``: iterated principal-axis factoring.
    ``rotation``: ``None``, ``"varimax"``, or ``"quartimax"`` (Kaiser normalization).
    """
    X = _as_2d(X, "X")
    n, p = X.shape
    if n_factors >= p:
        raise ValueError("n_factors must be less than the number of variables")
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1) if standardize else np.ones(p)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    R = np.cov(Z, rowvar=False, ddof=1) if not standardize else np.corrcoef(Z, rowvar=False)

    if method == "pa":
        psi = 1.0 - np.array([_smc(R, j) for j in range(p)])  # initial communality guess
        L = np.zeros((p, n_factors))
        for _ in range(max_iter):
            R_reduced = R - np.diag(psi) + np.diag(1.0 - psi)
            reduced = R.copy()
            np.fill_diagonal(reduced, 1.0 - psi)
            w, V = np.linalg.eigh(reduced)
            order = np.argsort(w)[::-1][:n_factors]
            w, V = np.clip(w[order], 0, None), V[:, order]
            L_new = V * np.sqrt(w)
            psi_new = np.clip(1.0 - np.sum(L_new ** 2, axis=1), 1e-4, 1.0)
            if np.max(np.abs(psi_new - psi)) < 1e-8:
                psi, L = psi_new, L_new
                break
            psi, L = psi_new, L_new
        loglik = None
    elif method == "ml":
        def neg_obj(log_psi):
            # Jöreskog's concentrated ML discrepancy: with L profiled out exactly (the
            # top n_factors eigenvalues of Psi^-1/2 R Psi^-1/2 are absorbed with zero
            # residual), only the p-n_factors smallest eigenvalues contribute:
            # F(Psi) = sum_{i>k} (w_i - log(w_i) - 1) >= 0, minimized over Psi.
            psi = np.exp(log_psi)
            Psi_inv_sqrt = np.diag(1.0 / np.sqrt(psi))
            M = Psi_inv_sqrt @ R @ Psi_inv_sqrt
            w = np.sort(np.linalg.eigvalsh(M))[::-1]
            w_rest = np.clip(w[n_factors:], 1e-10, None)
            return float(np.sum(w_rest - np.log(w_rest) - 1.0))

        x0 = np.log(np.clip(1.0 - np.array([_smc(R, j) for j in range(p)]), 0.05, 0.99))
        res = optimize.minimize(neg_obj, x0, method="L-BFGS-B",
                                 bounds=[(math.log(1e-4), math.log(1.0))] * p, options={"maxiter": max_iter})
        psi = np.exp(res.x)
        Psi_inv_sqrt = np.diag(1.0 / np.sqrt(psi))
        M = Psi_inv_sqrt @ R @ Psi_inv_sqrt
        w, V = np.linalg.eigh(M)
        order = np.argsort(w)[::-1][:n_factors]
        w, V = np.clip(w[order], 1.0, None), V[:, order]
        L = np.diag(np.sqrt(psi)) @ V @ np.diag(np.sqrt(np.clip(w - 1.0, 0, None)))
    else:
        raise ValueError("method must be 'ml' or 'pa'")

    if rotation in ("varimax", "quartimax"):
        L = _rotate(L, kind=rotation)

    Sigma = L @ L.T + np.diag(psi)
    sign, logdet = np.linalg.slogdet(Sigma)
    loglik = -0.5 * n * (p * math.log(2 * math.pi) + logdet + np.trace(np.linalg.solve(Sigma, R)))

    communalities = np.sum(L ** 2, axis=1)
    return FactorResult(loadings_=L, uniquenesses_=psi, communalities_=communalities,
                         loglik_=float(loglik), method=method, mean_=mean, scale_=scale)


def _smc(R, j):
    """Squared multiple correlation of variable j with the rest (initial communality)."""
    idx = [i for i in range(R.shape[0]) if i != j]
    R_jj = R[np.ix_(idx, idx)]
    r_j = R[idx, j]
    try:
        return float(r_j @ np.linalg.solve(R_jj, r_j))
    except np.linalg.LinAlgError:
        return 0.5


def _rotate(L, kind="varimax", max_iter=100, tol=1e-8):
    """Kaiser-normalized orthogonal rotation (varimax maximizes, quartimax is the
    gamma=0 special case of the same criterion)."""
    p, k = L.shape
    if k < 2:
        return L
    gamma = 1.0 if kind == "varimax" else 0.0
    h = np.sqrt(np.sum(L ** 2, axis=1))
    h = np.where(h > 0, h, 1.0)
    Ln = L / h[:, None]
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Z = Ln @ R
        u = Z ** 3 - (gamma / p) * Z @ np.diag(np.sum(Z ** 2, axis=0))
        U, s, Vt = np.linalg.svd(Ln.T @ u, full_matrices=False)
        R = U @ Vt
        d_new = np.sum(s)
        if abs(d_new - d) < tol:
            d = d_new
            break
        d = d_new
    return (Ln @ R) * h[:, None]


def canonical_correlation(X, Y):
    """Canonical correlation analysis between two variable sets via SVD of the
    whitened cross-covariance. ``.stats_`` holds Wilks' sequential-dimensionality
    tests (Bartlett's chi-squared approximation) for canonical correlations
    ``k, k+1, ..., min(p,q)`` being all zero, for ``k=0..min(p,q)-1``."""
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    n, p = X.shape
    q = Y.shape[1]
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)

    Sxx = Xc.T @ Xc / (n - 1)
    Syy = Yc.T @ Yc / (n - 1)
    Sxy = Xc.T @ Yc / (n - 1)

    Sxx_isqrt = _inv_sqrt(Sxx)
    Syy_isqrt = _inv_sqrt(Syy)
    K = Sxx_isqrt @ Sxy @ Syy_isqrt
    U, s, Vt = np.linalg.svd(K, full_matrices=False)
    s = np.clip(s, 0, 1.0)
    d = min(p, q)

    x_weights = Sxx_isqrt @ U[:, :d]
    y_weights = Syy_isqrt @ Vt.T[:, :d]
    x_scores = Xc @ x_weights
    y_scores = Yc @ y_weights

    stats = []
    for k in range(d):
        rem = s[k:] ** 2
        wilks = np.prod(1.0 - rem)
        df = (p - k) * (q - k)
        stat = -(n - 1 - 0.5 * (p + q + 1)) * math.log(max(wilks, 1e-300))
        from stochpylib.statistics._common import _chi2_sf
        pvalue = float(_chi2_sf(stat, df))
        stats.append({"k": k, "wilks": float(wilks), "chi2": float(stat), "df": df, "p": pvalue})

    return CanonicalCorrelationResult(correlations_=s[:d], x_weights_=x_weights, y_weights_=y_weights,
                                       x_scores_=x_scores, y_scores_=y_scores, stats_=stats)


def _inv_sqrt(S):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def discriminant_analysis(X, y, kind="lda", priors=None, reg=0.0):
    """Linear (``kind="lda"``) or quadratic (``kind="qda"``) discriminant analysis,
    by direct Gaussian Bayes classification (pooled covariance for LDA, per-class
    covariance for QDA; ``reg`` shrinks each class covariance toward its diagonal)."""
    X = _as_2d(X, "X")
    y = np.asarray(y)
    classes = np.unique(y)
    K = len(classes)
    n, p = X.shape
    means = np.array([X[y == c].mean(axis=0) for c in classes])
    ns = np.array([np.sum(y == c) for c in classes], dtype=float)
    pri = ns / n if priors is None else np.asarray(priors, dtype=float)

    covs = []
    for c in classes:
        Xc = X[y == c] - X[y == c].mean(axis=0)
        S = Xc.T @ Xc / (len(Xc) - 1) if len(Xc) > 1 else np.eye(p)
        S = (1 - reg) * S + reg * np.diag(np.diag(S))
        covs.append(S)
    covs = np.array(covs)

    if kind == "lda":
        pooled = np.sum((ns[:, None, None] - 1) * covs, axis=0) / (n - K)
        pooled_inv = np.linalg.pinv(pooled)
        coef = means @ pooled_inv.T
        intercept = -0.5 * np.sum((means @ pooled_inv) * means, axis=1) + np.log(pri)

        def discriminant(Xnew):
            Xnew = _as_2d(Xnew, "X")
            return Xnew @ coef.T + intercept

        def predict(Xnew):
            d = discriminant(Xnew)
            return classes[np.argmax(d, axis=1)]

        def predict_proba(Xnew):
            d = discriminant(Xnew)
            d -= d.max(axis=1, keepdims=True)
            w = np.exp(d)
            return w / w.sum(axis=1, keepdims=True)

        def transform(Xnew):
            Xnew = _as_2d(Xnew, "X")
            Sw_isqrt = _inv_sqrt(pooled)
            Xw = (Xnew - X.mean(axis=0)) @ Sw_isqrt
            Mw = (means - X.mean(axis=0)) @ Sw_isqrt
            Sb = (Mw * ns[:, None]).T @ Mw / n
            w, V = np.linalg.eigh(Sb)
            order = np.argsort(w)[::-1][: min(K - 1, p)]
            return Xw @ V[:, order]

        res = DiscriminantResult(kind="lda", classes_=classes, priors_=pri, means_=means,
                                  coef_=coef, intercept_=intercept)
    else:
        def log_gauss(Xnew, mean, cov):
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(p))
            diff = (Xnew - mean) @ np.linalg.inv(L).T
            maha = np.sum(diff ** 2, axis=1)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            return -0.5 * (maha + logdet + p * math.log(2 * math.pi))

        def discriminant(Xnew):
            Xnew = _as_2d(Xnew, "X")
            return np.column_stack([log_gauss(Xnew, means[k], covs[k]) + math.log(pri[k]) for k in range(K)])

        def predict(Xnew):
            return classes[np.argmax(discriminant(Xnew), axis=1)]

        def predict_proba(Xnew):
            d = discriminant(Xnew)
            d -= d.max(axis=1, keepdims=True)
            w = np.exp(d)
            return w / w.sum(axis=1, keepdims=True)

        def transform(Xnew):
            raise NotImplementedError("transform() is only defined for LDA")

        res = DiscriminantResult(kind="qda", classes_=classes, priors_=pri, means_=means,
                                  coef_=None, intercept_=None)

    res.extras["_predict"] = predict
    res.extras["_predict_proba"] = predict_proba
    res.extras["_transform"] = transform
    res.extras["covariances"] = covs
    return res


def cluster_analysis(X, n_clusters, method="kmeans", linkage="ward", n_init=10,
                      max_iter=300, random_state=None):
    """K-means (``method="kmeans"``, k-means++ init + Lloyd's algorithm) or
    agglomerative hierarchical clustering (``method="hierarchical"``,
    ``linkage`` in ward/single/complete/average, Lance-Williams update)."""
    X = _as_2d(X, "X")
    n = X.shape[0]
    rng = _rng(random_state)

    if method == "kmeans":
        best_labels, best_centers, best_inertia = None, None, np.inf
        for _ in range(n_init):
            centers = _kmeans_pp_init(X, n_clusters, rng)
            labels = np.zeros(n, dtype=int)
            for _ in range(max_iter):
                d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                new_labels = np.argmin(d2, axis=1)
                if np.array_equal(new_labels, labels) and _ > 0:
                    labels = new_labels
                    break
                labels = new_labels
                for k in range(n_clusters):
                    if np.any(labels == k):
                        centers[k] = X[labels == k].mean(axis=0)
            inertia = float(np.sum((X - centers[labels]) ** 2))
            if inertia < best_inertia:
                best_labels, best_centers, best_inertia = labels, centers, inertia
        labels, centers, inertia = best_labels, best_centers, best_inertia
        Z = None
    elif method == "hierarchical":
        Z, labels = _agglomerative(X, n_clusters, linkage)
        centers = np.array([X[labels == k].mean(axis=0) for k in np.unique(labels)])
        inertia = float(np.sum((X - centers[labels]) ** 2))
    else:
        raise ValueError("method must be 'kmeans' or 'hierarchical'")

    sil = _silhouette(X, labels) if n_clusters > 1 and n_clusters < n else float("nan")
    return ClusterResult(labels_=labels, method=method, centers_=centers, inertia_=inertia,
                          linkage_=Z, silhouette_=sil, n_clusters=n_clusters)


def _kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    idx0 = rng.integers(n)
    centers[0] = X[idx0]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centers[i]) ** 2, axis=1))
    return centers


def _agglomerative(X, n_clusters, linkage):
    """Lance-Williams agglomerative clustering; returns a scipy-format linkage
    matrix Z and the final n_clusters-way cut labels."""
    n = X.shape[0]
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    active = list(range(n))
    size = {i: 1 for i in range(n)}
    cluster_id = {i: i for i in range(n)}
    next_id = n
    Z = []
    dist = {(min(i, j), max(i, j)): D[i, j] for i in range(n) for j in range(i + 1, n)}

    while len(active) > 1:
        (a, b), d_ab = min(dist.items(), key=lambda kv: kv[1])
        active.remove(a)
        active.remove(b)
        new_cluster = next_id
        ni, nj = size[a], size[b]
        for c in active:
            d_ac = dist.get((min(a, c), max(a, c)))
            d_bc = dist.get((min(b, c), max(b, c)))
            nk = size[c]
            if linkage == "single":
                d_new = min(d_ac, d_bc)
            elif linkage == "complete":
                d_new = max(d_ac, d_bc)
            elif linkage == "average":
                d_new = (ni * d_ac + nj * d_bc) / (ni + nj)
            elif linkage == "ward":
                d_new = math.sqrt(((ni + nk) * d_ac ** 2 + (nj + nk) * d_bc ** 2 - nk * d_ab ** 2) / (ni + nj + nk))
            else:
                raise ValueError("linkage must be 'ward', 'single', 'complete', or 'average'")
            dist[(min(new_cluster, c), max(new_cluster, c))] = d_new
            del dist[(min(a, c), max(a, c))]
            del dist[(min(b, c), max(b, c))]
        del dist[(a, b)]
        Z.append([cluster_id[a], cluster_id[b], d_ab, ni + nj])
        size[new_cluster] = ni + nj
        cluster_id[new_cluster] = next_id
        active.append(new_cluster)
        next_id += 1

    Z = np.array(Z)
    labels = _cut_tree(Z, n, n_clusters)
    return Z, labels


def _cut_tree(Z, n, n_clusters):
    """Cut a scipy-format linkage matrix into n_clusters flat labels."""
    parent = {i: i for i in range(2 * n - 1)}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    merges_to_do = n - n_clusters
    for i in range(merges_to_do):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        parent[find(a)] = n + i
        parent[find(b)] = n + i
    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


def _silhouette(X, labels):
    n = X.shape[0]
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    clusters = np.unique(labels)
    s = np.zeros(n)
    for i in range(n):
        own = labels[i]
        in_mask = (labels == own)
        a = D[i, in_mask].sum() / max(in_mask.sum() - 1, 1)
        b = np.inf
        for c in clusters:
            if c == own:
                continue
            mask = labels == c
            if mask.sum() == 0:
                continue
            b = min(b, D[i, mask].mean())
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(np.mean(s))


def MDS(D, n_components=2, method="classical", metric=True, max_iter=300, random_state=None):
    """Multidimensional scaling. ``method="classical"``: Torgerson double-centering
    eigendecomposition (exact for a Euclidean distance matrix). ``method="smacof"``:
    Guttman-transform iterative stress majorization (metric, or non-metric with an
    isotonic-regression disparity step when ``metric=False``)."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D must be a square distance matrix")

    if method == "classical":
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        w, V = np.linalg.eigh(B)
        order = np.argsort(w)[::-1][:n_components]
        w_top, V_top = np.clip(w[order], 0, None), V[:, order]
        embedding = V_top * np.sqrt(w_top)
        dist_fit = np.sqrt(np.sum((embedding[:, None, :] - embedding[None, :, :]) ** 2, axis=2))
        iu = np.triu_indices(n, 1)
        stress = float(np.sqrt(np.sum((D[iu] - dist_fit[iu]) ** 2) / np.sum(D[iu] ** 2)))
        return MDSResult(embedding_=embedding, stress_=stress, method="classical", eigenvalues_=w[order])

    if method == "smacof":
        rng = _rng(random_state)
        embedding = rng.normal(size=(n, n_components)) * D.max() / math.sqrt(n_components)
        iu = np.triu_indices(n, 1)
        disparities = D.copy()
        prev_stress = np.inf
        for it in range(max_iter):
            dist_fit = np.sqrt(np.sum((embedding[:, None, :] - embedding[None, :, :]) ** 2, axis=2))
            if not metric:
                disparities = D.copy()
                order = np.argsort(D[iu])
                pooled = _pava(dist_fit[iu][order])
                disparities[iu] = np.empty_like(pooled)
                disparities[iu][order] = pooled
                disparities.T[iu] = disparities[iu]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(dist_fit > 1e-12, disparities / dist_fit, 0.0)
            np.fill_diagonal(ratio, 0.0)
            B = -ratio  # Guttman transform matrix: B_ij = -w_ij*disparity_ij/dist_ij (j!=i)
            B[np.diag_indices(n)] = np.sum(ratio, axis=1)  # B_ii = -sum_{j!=i} B_ij
            embedding = (B @ embedding) / n
            stress = float(np.sum((disparities[iu] - dist_fit[iu]) ** 2))
            if abs(prev_stress - stress) < 1e-9 * max(prev_stress, 1.0):
                prev_stress = stress
                break
            prev_stress = stress
        stress_norm = float(math.sqrt(prev_stress / np.sum(disparities[iu] ** 2))) if np.sum(disparities[iu] ** 2) > 0 else 0.0
        return MDSResult(embedding_=embedding, stress_=stress_norm, method="smacof", n_iter_=it + 1)

    raise ValueError("method must be 'classical' or 'smacof'")


def _pava(y):
    """Pool-adjacent-violators isotonic regression (non-decreasing fit)."""
    y = np.asarray(y, dtype=float)
    stack = []
    for v in y:
        stack.append([v, 1])
        while len(stack) > 1 and stack[-2][0] > stack[-1][0]:
            v2, w2 = stack.pop()
            v1, w1 = stack.pop()
            stack.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    result = []
    for val, w in stack:
        result.extend([val] * int(w))
    return np.array(result)
