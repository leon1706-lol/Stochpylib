"""Special-purpose plots that need a distinctive layout: Markov-chain diagrams,
Brownian-motion fans, GP posterior bands, Kaplan-Meier curves, variograms and
eigenvalue spectra.
"""

import numpy as np

from stochpylib.viz._common import _as_1d, _get_fig_ax, _rng

__all__ = ["plot_markov_chain", "plot_brownian", "plot_gp", "plot_survival_km",
           "plot_variogram", "plot_eigenvalues"]


def _extract_transition_matrix(obj):
    if isinstance(obj, (np.ndarray, list, tuple)):
        P = np.asarray(obj, dtype=float)
    elif hasattr(obj, "transition_matrix_"):
        P = np.asarray(obj.transition_matrix_, dtype=float)
    elif hasattr(obj, "transition_"):
        P = np.asarray(obj.transition_, dtype=float)
    else:
        raise ValueError("could not find a transition matrix on this object")
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("transition matrix must be square")
    return P


def plot_markov_chain(P, states=None, threshold=0.01, ax=None, title=None):
    """Directed graph of a Markov chain's transition matrix, node size ~ sqrt(stationary
    probability), edges labelled with their probability."""
    fig, axes = _get_fig_ax(ax, width=520, height=520)
    P = _extract_transition_matrix(P)
    k = P.shape[0]
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("transition matrix rows must sum to 1")
    labels = list(states) if states is not None else [str(i) for i in range(k)]

    # Stationary distribution: left eigenvector of eigenvalue 1, power iteration (robust
    # for irreducible/aperiodic chains without needing complex eigen-decomposition).
    pi = np.full(k, 1.0 / k)
    for _ in range(2000):
        pi_next = pi @ P
        if np.max(np.abs(pi_next - pi)) < 1e-14:
            pi = pi_next
            break
        pi = pi_next
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()

    angles = 2 * np.pi * np.arange(k) / k
    radius = 1.0
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    node_r = 0.08 + 0.12 * np.sqrt(pi / (pi.max() or 1.0))
    for i in range(k):
        axes.circle(xs[i], ys[i], node_r[i], color="#4E79A7")
        # label below the node, not inside it -- a long state name would otherwise
        # overflow a small (low-stationary-probability) node's circle
        axes.text(xs[i], ys[i] - node_r[i] - 0.06, labels[i], anchor="middle", size=10,
                  color="#000000")
        if P[i, i] >= threshold:  # self-loop: a small circle tangent to the node's top
            loop_r = 0.05
            axes.circle(xs[i], ys[i] + node_r[i] + loop_r, loop_r, color="#999999", fill=False)
            axes.text(xs[i], ys[i] + node_r[i] + 2 * loop_r + 0.05, f"{P[i, i]:.2f}",
                      anchor="middle", size=8, color="#333333")
    for i in range(k):
        for j in range(k):
            if i == j or P[i, j] < threshold:
                continue
            curve = 0.25 if P[j, i] >= threshold else 0.08
            axes.arrow(xs[i], ys[i], xs[j], ys[j], curved=curve, color="#999999", width=1.0)
            # label near the source end (not the shared midpoint), so the i->j and j->i
            # labels for a bidirectional pair don't land on top of each other
            t_frac = 0.3
            lx = xs[i] + t_frac * (xs[j] - xs[i])
            ly = ys[i] + t_frac * (ys[j] - ys[i])
            axes.text(lx, ly, f"{P[i, j]:.2f}", anchor="middle", size=8, color="#333333")
    axes.set(title=title, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", grid=False)
    fig.data = {"P": P, "stationary": pi, "node_xy": np.column_stack([xs, ys])}
    return fig


def plot_brownian(n_paths=5, n_steps=500, T=1.0, mu=0.0, sigma=1.0, x0=0.0,
                  geometric=False, envelope=True, random_state=None, ax=None, title=None):
    """Sample paths of (geometric) Brownian motion, with an analytic envelope."""
    fig, axes = _get_fig_ax(ax)
    rng = _rng(random_state)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    increments = mu * dt + sigma * dW
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    paths = np.exp(x0 + W) if geometric else x0 + W

    for i in range(n_paths):
        axes.line(t, paths[i], alpha=0.7 if n_paths > 1 else 1.0)
    out = {"t": t, "paths": paths}
    if envelope:
        if geometric:
            m = x0 + mu * t
            s = sigma * np.sqrt(t)
            lower = np.exp(m - 1.96 * s)
            upper = np.exp(m + 1.96 * s)
        else:
            lower = x0 + mu * t - 2.0 * sigma * np.sqrt(t)
            upper = x0 + mu * t + 2.0 * sigma * np.sqrt(t)
        axes.band(t, lower, upper, color="#999999", label="envelope")
        out["lower"], out["upper"] = lower, upper
    axes.set(title=title, xlabel="t", ylabel="X(t)")
    fig.data = out
    return fig


def plot_gp(gp, X=None, X_train=None, y_train=None, level=0.95, n_samples=0,
           random_state=None, ax=None, title=None):
    """A fitted 1-D Gaussian process's posterior mean and confidence band, with the
    training points and (optionally) posterior sample draws."""
    from stochpylib.statistics._common import _norm_ppf

    fig, axes = _get_fig_ax(ax)
    if X is None:
        Xt = np.asarray(gp.X_train, dtype=float)
        X = np.linspace(Xt[:, 0].min(), Xt[:, 0].max(), 200)[:, None]
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    z = _norm_ppf(0.5 + level / 2.0)
    if n_samples > 0:
        mean, cov = gp.predict(X, full_cov=True)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(X)))
        draws = mean[:, None] + L @ _rng(random_state).standard_normal((len(X), n_samples))
        for k in range(n_samples):
            axes.line(X[:, 0], draws[:, k], color="#B07AA1", alpha=0.4,
                      label="posterior sample" if k == 0 else None)
    else:
        mean, std = gp.predict(X, return_std=True)
        draws = None

    lower, upper = mean - z * std, mean + z * std
    axes.band(X[:, 0], lower, upper, label=f"{int(level * 100)}% band")
    axes.line(X[:, 0], mean, color="#4E79A7", width=2.0, label="mean")
    Xtr = X_train if X_train is not None else getattr(gp, "X_train", None)
    ytr = y_train if y_train is not None else getattr(gp, "y_train", None)
    if Xtr is not None and ytr is not None:
        Xtr = np.asarray(Xtr, dtype=float)
        axes.scatter(Xtr[:, 0] if Xtr.ndim > 1 else Xtr, np.asarray(ytr, dtype=float),
                    color="#E15759", label="training data")
    axes.set(title=title, xlabel="x", ylabel="f(x)", legend="best")
    out = {"x": X[:, 0], "mean": mean, "lower": lower, "upper": upper}
    if draws is not None:
        out["samples"] = draws
    fig.data = out
    return fig


def plot_survival_km(km_or_durations, events=None, groups=None, ci=True, censor_marks=True,
                     ax=None, title=None):
    """Kaplan-Meier survival curve(s), with confidence bands and censoring ticks;
    multiple groups get a log-rank test in the title."""
    from stochpylib.survival import KaplanMeier

    fig, axes = _get_fig_ax(ax)
    # censored_by_name[name] holds (censored_times, survival_at_those_times), or None when
    # only a pre-fitted KaplanMeier was given (no raw per-subject data to mark censoring on).
    censored_by_name = {}
    if hasattr(km_or_durations, "survival_function_"):
        fits = {"": km_or_durations}
        censored_by_name[""] = None
    else:
        durations = _as_1d(km_or_durations, "durations")
        events_arr = np.asarray(events, dtype=float).ravel() if events is not None else \
            np.ones_like(durations)
        if groups is not None:
            groups_arr = np.asarray(groups)
            fits = {}
            for g in np.unique(groups_arr):
                mask = groups_arr == g
                name = str(g)
                fits[name] = KaplanMeier().fit(durations[mask], events_arr[mask])
                censored_by_name[name] = durations[mask][events_arr[mask] == 0]
        else:
            fits = {"": KaplanMeier().fit(durations, events_arr)}
            censored_by_name[""] = durations[events_arr == 0]

    out = {}
    for name, km in fits.items():
        sf = km.survival_function_
        t, s = sf["time"], sf["value"]
        if t[0] > 0:  # conventional KM start: S(0) = 1 held until the first event
            t = np.concatenate([[0.0], t])
            s = np.concatenate([[1.0], s])
        step_artist = axes.step(t, s, where="post", label=name or "KM")
        if ci and km.confidence_interval_ is not None:
            ci_arr = km.confidence_interval_
            ci_t, ci_lo, ci_hi = ci_arr["time"], ci_arr["lower"], ci_arr["upper"]
            if ci_t[0] > 0:
                ci_t = np.concatenate([[0.0], ci_t])
                ci_lo = np.concatenate([[1.0], ci_lo])
                ci_hi = np.concatenate([[1.0], ci_hi])
            axes.band(ci_t, ci_lo, ci_hi, color=step_artist.color, alpha=0.2)
        censored = censored_by_name.get(name)
        if censor_marks and censored is not None and len(censored):
            censored_s = km.predict(censored)
            axes.scatter(censored, censored_s, size=4.0, marker="+", color=step_artist.color)
        out[name or "KM"] = {"time": t, "survival": s}

    subtitle = title
    if groups is not None and len(fits) > 1:
        from stochpylib.survival import LogRankTest

        lr = LogRankTest().fit(durations, events_arr, groups)
        subtitle = (title or "Kaplan-Meier") + f" (log-rank p={lr.p_value_:.3g})"
    axes.set(title=subtitle, xlabel="time", ylabel="S(t)", ylim=(0.0, 1.0),
             legend="best" if len(fits) > 1 else None)
    fig.data = out
    return fig


def plot_variogram(obj, model=None, coords=None, values=None, bins=15, ax=None, title=None):
    """Experimental (semi)variogram points plus a fitted model curve, spatial-covariance
    curve, or a spatial summary function (Ripley's K / pair correlation) with its envelope."""
    fig, axes = _get_fig_ax(ax)
    if hasattr(obj, "r") and hasattr(obj, "estimate"):  # SpatialFunction
        axes.scatter(obj.r, obj.estimate, label=obj.name or "estimate")
        axes.line(obj.r, obj.theoretical, color="#E15759", label="theoretical (CSR)")
        if obj.lower is not None:
            axes.band(obj.r, obj.lower, obj.upper, label="envelope")
        axes.set(title=title, xlabel="r", ylabel=obj.name or "value", legend="best")
        fig.data = {"r": obj.r, "estimate": obj.estimate, "theoretical": obj.theoretical}
        return fig

    if hasattr(obj, "lags_"):  # ExperimentalVariogram (already fitted)
        exp_vario = obj
    elif coords is not None:
        from stochpylib.spatial_statistics import ExperimentalVariogram

        exp_vario = ExperimentalVariogram(bins=bins).fit(coords, values)
    else:
        exp_vario = None

    out = {}
    if exp_vario is not None:
        sizes = 3.0 + 6.0 * (exp_vario.counts_ / (exp_vario.counts_.max() or 1))
        for lag, gamma, size in zip(exp_vario.lags_, exp_vario.gamma_, sizes):
            axes.scatter(np.array([lag]), np.array([gamma]), size=float(size), color="#4E79A7")
        out.update(lags=exp_vario.lags_, gamma=exp_vario.gamma_, counts=exp_vario.counts_)

    # A model curve can come from `model=` (a callable Semivariogram/SpatialCovariance, or
    # a fitted VariogramFitting via its `.variogram_`), or from `obj` itself when it is one
    # of those instead of an ExperimentalVariogram/data source.
    candidate = model if model is not None else (None if exp_vario is not None else obj)
    fitted = candidate.variogram_ if hasattr(candidate, "variogram_") else candidate
    if fitted is not None and callable(fitted):
        h_max = exp_vario.lags_.max() if exp_vario is not None else 10.0
        h = np.linspace(1e-6, h_max, 200)
        curve = np.asarray([fitted(hi) for hi in h], dtype=float)
        axes.line(h, curve, color="#E15759", label="model")
        out["model_h"], out["model_gamma"] = h, curve

    axes.set(title=title, xlabel="lag distance", ylabel="semivariance", legend="best")
    fig.data = out
    return fig


def plot_eigenvalues(obj, law="auto", kind="auto", bins=50, random_state=None, ax=None,
                     title=None):
    """Empirical eigenvalue spectrum vs. its limiting law: histogram + semicircle/
    Marchenko-Pastur density for real spectra, unit-circle scatter for complex ones,
    or nearest-neighbour spacing vs. the Wigner surmise (``kind="spacing"``)."""
    fig, axes = _get_fig_ax(ax)
    if hasattr(obj, "eigenvalues"):
        eig = np.asarray(obj.eigenvalues(random_state=random_state))
        ensemble = obj
    else:
        arr = np.asarray(obj)
        if arr.ndim == 2:
            eig = np.linalg.eigvals(arr)
        else:
            eig = arr
        ensemble = None

    is_complex = np.iscomplexobj(eig) and np.max(np.abs(eig.imag)) > 1e-9 * (
        np.max(np.abs(eig.real)) or 1.0)

    if kind == "spacing":
        from stochpylib.random_matrix.statistics import EigenvalueSpacing

        spacing = EigenvalueSpacing(eig)
        s = spacing.spacings()
        heights, edges = np.histogram(s, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes.bar(centers, heights, width=np.diff(edges), label="empirical")
        beta = 1
        wigner = (np.pi / 2.0) * centers * np.exp(-np.pi / 4.0 * centers ** 2)
        axes.line(centers, wigner, color="#E15759", label="Wigner surmise")
        axes.set(title=title, xlabel="spacing", ylabel="density", legend="best")
        fig.data = {"centers": centers, "empirical": heights, "theoretical": wigner}
        return fig

    if is_complex and kind == "auto":
        axes.scatter(eig.real, eig.imag, size=3.0)
        theta = np.linspace(0, 2 * np.pi, 200)
        axes.line(np.cos(theta), np.sin(theta), color="#E15759", label="unit circle")
        axes.set(title=title, xlabel="Re", ylabel="Im", aspect="equal", legend="best")
        fig.data = {"eigenvalues": eig}
        return fig

    eig_real = eig.real if np.iscomplexobj(eig) else eig
    if law == "auto":
        if ensemble is not None and hasattr(ensemble, "limit_law"):
            law_obj = ensemble.limit_law()
            plotted = ensemble.normalize(eig_real) if hasattr(ensemble, "normalize") else eig_real
        else:
            raise ValueError("law='auto' needs an ensemble with a limit_law(); pass one explicitly")
    else:
        law_obj = law
        plotted = eig_real
    centers, empirical, theoretical = law_obj.histogram_vs_density(plotted, bins=bins)
    axes.bar(centers, empirical, width=np.diff(np.append(centers, centers[-1] +
             (centers[1] - centers[0] if len(centers) > 1 else 1.0))), label="empirical")
    axes.line(centers, theoretical, color="#E15759", label=type(law_obj).__name__)
    axes.set(title=title, xlabel="eigenvalue", ylabel="density", legend="best")
    fig.data = {"centers": centers, "empirical": empirical, "theoretical": theoretical}
    return fig
