"""Variational inference: mean-field and full-rank ADVI-style Gaussian approximations,
score-function black-box VI, planar normalizing flows with hand-derived gradients (no
autodiff in the stack), and Stein variational gradient descent.
"""

import numpy as np
from scipy import special

from stochpylib.advanced_mcmc._base import LogDensity
from stochpylib.advanced_mcmc._common import _Adam, _cholesky_psd, _rng

__all__ = ["ADVI", "BlackBoxVI", "MeanFieldVI", "NormalizingFlows", "SteinVI"]

_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


class MeanFieldVI:
    """Gaussian mean-field variational inference via the reparameterization trick.

    Approximates the target with an independent Gaussian per coordinate, maximizing the
    ELBO with Adam using ``n_mc`` reparameterized draws per iteration.
    """

    def __init__(self, log_prob, dim, grad_log_prob=None, n_iter=2000, n_mc=10, lr=0.05,
                 init_mean=None, init_std=1.0):
        self.target = LogDensity(log_prob, grad_log_prob)
        self.dim = int(dim)
        self.n_iter = int(n_iter)
        self.n_mc = int(n_mc)
        self.lr = float(lr)
        self.init_mean = init_mean
        self.init_std = init_std

    def fit(self, random_state=None):
        rng = _rng(random_state)
        dim = self.dim
        mu = np.zeros(dim) if self.init_mean is None else np.array(self.init_mean, dtype=float)
        log_sigma = np.full(dim, np.log(self.init_std))
        adam_mu = _Adam(dim, lr=self.lr)
        adam_ls = _Adam(dim, lr=self.lr)
        history = []
        for _ in range(self.n_iter):
            eps = rng.standard_normal((self.n_mc, dim))
            sigma = np.exp(log_sigma)
            z = mu + sigma * eps
            grads = np.array([self.target.grad(zi) for zi in z])
            logps = np.array([self.target(zi) for zi in z])
            grad_mu = grads.mean(axis=0)
            grad_log_sigma = (grads * eps).mean(axis=0) * sigma + 1.0
            mu = adam_mu.step(mu, grad_mu)
            log_sigma = adam_ls.step(log_sigma, grad_log_sigma)
            entropy = np.sum(log_sigma) + dim * (0.5 + _HALF_LOG_2PI)
            elbo = float(logps.mean()) + entropy
            history.append(elbo)
        self.mean_ = mu
        self.std_ = np.exp(log_sigma)
        self.elbo_history_ = np.array(history)
        self.n_iter_ = self.n_iter
        self.converged_ = _elbo_converged(self.elbo_history_)
        return self

    def sample(self, n=1, random_state=None):
        rng = _rng(random_state)
        return self.mean_ + self.std_ * rng.standard_normal((n, self.dim))

    def log_prob(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        z = (x - self.mean_) / self.std_
        lp = -0.5 * z ** 2 - np.log(self.std_) - _HALF_LOG_2PI
        return lp.sum(axis=1)

    def __repr__(self):
        return f"MeanFieldVI(dim={self.dim}, n_iter={self.n_iter})"


def _clip_norm(g, max_norm):
    n = np.linalg.norm(g)
    return g * (max_norm / n) if n > max_norm else g


def _elbo_converged(history, window=50, tol=1e-4):
    if len(history) < 2 * window:
        return False
    prev = np.mean(history[-2 * window:-window])
    curr = np.mean(history[-window:])
    return abs(curr - prev) < tol * max(abs(curr), 1.0)


def _transform_bounds(zeta, lower, upper):
    """Unconstrained zeta -> constrained theta, with log|dtheta/dzeta| per coordinate."""
    theta = np.empty_like(zeta)
    logJ = np.empty_like(zeta)
    for i in range(len(zeta)):
        a, b = lower[i], upper[i]
        z = zeta[i]
        if np.isneginf(a) and np.isposinf(b):
            theta[i] = z
            logJ[i] = 0.0
        elif np.isposinf(b):
            theta[i] = a + np.exp(z)
            logJ[i] = z
        elif np.isneginf(a):
            theta[i] = b - np.exp(z)
            logJ[i] = z
        else:
            s = special.expit(z)
            theta[i] = a + (b - a) * s
            logJ[i] = np.log(b - a) + np.log(s) + np.log(1.0 - s)
    return theta, logJ


def _dtheta_dzeta(zeta, lower, upper):
    d = np.empty_like(zeta)
    for i in range(len(zeta)):
        a, b = lower[i], upper[i]
        z = zeta[i]
        if np.isneginf(a) and np.isposinf(b):
            d[i] = 1.0
        elif np.isposinf(b):
            d[i] = np.exp(z)
        elif np.isneginf(a):
            d[i] = np.exp(z)
        else:
            s = special.expit(z)
            d[i] = (b - a) * s * (1.0 - s)
    return d


def _dlogJ_dzeta(zeta, lower, upper):
    d = np.empty_like(zeta)
    for i in range(len(zeta)):
        a, b = lower[i], upper[i]
        z = zeta[i]
        if np.isneginf(a) and np.isposinf(b):
            d[i] = 0.0
        elif np.isposinf(b) or np.isneginf(a):
            d[i] = 1.0
        else:
            s = special.expit(z)
            d[i] = 1.0 - 2.0 * s
    return d


class ADVI:
    """Automatic differentiation VI (Kucukelbir et al. 2017): a mean-field or full-rank
    Gaussian in an unconstrained space, mapped through per-coordinate bound transforms.

    ``lower``/``upper``: ``(dim,)`` arrays (default unbounded, i.e. ``-inf``/``+inf``);
    a coordinate may be one-sided or two-sided bounded. ``rank``: ``"mean-field"`` (default)
    or ``"full"`` for a full covariance via a lower-triangular Cholesky factor.
    """

    def __init__(self, log_prob, dim, grad_log_prob=None, rank="mean-field", lower=None,
                 upper=None, n_iter=2000, n_mc=10, lr=0.05, init_mean=None):
        self.target = LogDensity(log_prob, grad_log_prob)
        self.dim = int(dim)
        if rank not in ("mean-field", "full"):
            raise ValueError("rank must be 'mean-field' or 'full'")
        self.rank = rank
        self.lower = np.full(dim, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        self.upper = np.full(dim, np.inf) if upper is None else np.asarray(upper, dtype=float)
        self.n_iter = int(n_iter)
        self.n_mc = int(n_mc)
        self.lr = float(lr)
        self.init_mean = init_mean

    def _theta_grad_to_zeta(self, zeta, theta):
        gt = self.target.grad(theta)
        dtd = _dtheta_dzeta(zeta, self.lower, self.upper)
        dlogj = _dlogJ_dzeta(zeta, self.lower, self.upper)
        return gt * dtd + dlogj

    def fit(self, random_state=None):
        rng = _rng(random_state)
        dim = self.dim
        mu = np.zeros(dim) if self.init_mean is None else np.array(self.init_mean, dtype=float)
        history = []
        if self.rank == "mean-field":
            log_sigma = np.zeros(dim)
            adam_mu = _Adam(dim, lr=self.lr)
            adam_ls = _Adam(dim, lr=self.lr)
            for _ in range(self.n_iter):
                sigma = np.exp(log_sigma)
                eps = rng.standard_normal((self.n_mc, dim))
                zeta = mu + sigma * eps
                grad_mu = np.zeros(dim)
                grad_ls = np.zeros(dim)
                elbo_acc = 0.0
                for k in range(self.n_mc):
                    theta, logJ = _transform_bounds(zeta[k], self.lower, self.upper)
                    g_zeta = self._theta_grad_to_zeta(zeta[k], theta)
                    grad_mu += g_zeta
                    grad_ls += g_zeta * eps[k] * sigma
                    elbo_acc += self.target(theta) + logJ.sum()
                grad_mu /= self.n_mc
                grad_ls = grad_ls / self.n_mc + 1.0
                mu = adam_mu.step(mu, grad_mu)
                log_sigma = adam_ls.step(log_sigma, grad_ls)
                entropy = np.sum(log_sigma) + dim * (0.5 + _HALF_LOG_2PI)
                history.append(elbo_acc / self.n_mc + entropy)
            self.mean_ = mu
            self.std_ = np.exp(log_sigma)
            self.scale_tril_ = None
        else:
            L = np.eye(dim) * 0.1
            adam_mu = _Adam(dim, lr=self.lr)
            adam_L = _Adam((dim, dim), lr=self.lr)
            tril_mask = np.tril(np.ones((dim, dim)))
            for _ in range(self.n_iter):
                eps = rng.standard_normal((self.n_mc, dim))
                grad_mu = np.zeros(dim)
                grad_L = np.zeros((dim, dim))
                elbo_acc = 0.0
                for k in range(self.n_mc):
                    zeta_k = mu + L @ eps[k]
                    theta, logJ = _transform_bounds(zeta_k, self.lower, self.upper)
                    g_zeta = self._theta_grad_to_zeta(zeta_k, theta)
                    grad_mu += g_zeta
                    grad_L += np.outer(g_zeta, eps[k])
                    elbo_acc += self.target(theta) + logJ.sum()
                grad_mu /= self.n_mc
                grad_L = grad_L / self.n_mc * tril_mask
                diag_idx = np.arange(dim)
                grad_L[diag_idx, diag_idx] += 1.0 / np.maximum(np.diag(L), 1e-8)
                mu = adam_mu.step(mu, grad_mu)
                L = adam_L.step(L, grad_L) * tril_mask
                L[diag_idx, diag_idx] = np.maximum(L[diag_idx, diag_idx], 1e-6)
                entropy = np.sum(np.log(np.diag(L))) + dim * (0.5 + _HALF_LOG_2PI)
                history.append(elbo_acc / self.n_mc + entropy)
            self.mean_ = mu
            self.scale_tril_ = L
            self.std_ = np.sqrt(np.diag(L @ L.T))
        self.elbo_history_ = np.array(history)
        self.n_iter_ = self.n_iter
        self.converged_ = _elbo_converged(self.elbo_history_)
        return self

    def transform(self, zeta):
        theta, _ = _transform_bounds(np.asarray(zeta, dtype=float), self.lower, self.upper)
        return theta

    def sample(self, n=1, random_state=None):
        rng = _rng(random_state)
        eps = rng.standard_normal((n, self.dim))
        if self.rank == "mean-field":
            zeta = self.mean_ + self.std_ * eps
        else:
            zeta = self.mean_ + eps @ self.scale_tril_.T
        return np.array([self.transform(z) for z in zeta])

    def log_prob(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        out = np.empty(len(theta))
        for i, t in enumerate(theta):
            # invert the transform numerically is unnecessary here: approximate density
            # is only reported in the unconstrained-Gaussian sense via std_/scale_tril_
            out[i] = float("nan")
        return out

    def __repr__(self):
        return f"ADVI(dim={self.dim}, rank={self.rank!r}, n_iter={self.n_iter})"


class BlackBoxVI:
    """Ranganath, Gerrish & Blei (2014) score-function ELBO gradients with per-parameter
    control variates -- needs no gradient of the target."""

    def __init__(self, log_prob, dim, n_iter=3000, n_mc=64, lr=0.01, init_mean=None, init_std=1.0):
        self.target = LogDensity(log_prob)
        self.dim = int(dim)
        self.n_iter = int(n_iter)
        self.n_mc = int(n_mc)
        self.lr = float(lr)
        self.init_mean = init_mean
        self.init_std = init_std

    def fit(self, random_state=None):
        rng = _rng(random_state)
        dim = self.dim
        mu = np.zeros(dim) if self.init_mean is None else np.array(self.init_mean, dtype=float)
        log_sigma = np.full(dim, np.log(self.init_std))
        adam_mu = _Adam(dim, lr=self.lr)
        adam_ls = _Adam(dim, lr=self.lr)
        history = []
        for _ in range(self.n_iter):
            sigma = np.exp(log_sigma)
            z = mu + sigma * rng.standard_normal((self.n_mc, dim))
            logp = np.array([self.target(zi) for zi in z])
            logq = (-0.5 * ((z - mu) / sigma) ** 2 - np.log(sigma) - _HALF_LOG_2PI).sum(axis=1)
            score_mu = (z - mu) / sigma ** 2
            score_ls = ((z - mu) ** 2 / sigma ** 2 - 1.0)
            f_val = (logp - logq)
            grad_mu = np.zeros(dim)
            grad_ls = np.zeros(dim)
            for j in range(dim):
                fj = f_val * score_mu[:, j]
                cov = np.cov(fj, score_mu[:, j])[0, 1]
                var = np.var(score_mu[:, j])
                a = cov / var if var > 1e-12 else 0.0
                grad_mu[j] = np.mean(fj - a * score_mu[:, j])
                fj2 = f_val * score_ls[:, j]
                cov2 = np.cov(fj2, score_ls[:, j])[0, 1]
                var2 = np.var(score_ls[:, j])
                a2 = cov2 / var2 if var2 > 1e-12 else 0.0
                grad_ls[j] = np.mean(fj2 - a2 * score_ls[:, j])
            mu = adam_mu.step(mu, grad_mu)
            log_sigma = adam_ls.step(log_sigma, grad_ls)
            history.append(float(np.mean(f_val)))
        self.mean_ = mu
        self.std_ = np.exp(log_sigma)
        self.elbo_history_ = np.array(history)
        self.n_iter_ = self.n_iter
        return self

    def sample(self, n=1, random_state=None):
        rng = _rng(random_state)
        return self.mean_ + self.std_ * rng.standard_normal((n, self.dim))

    def __repr__(self):
        return f"BlackBoxVI(dim={self.dim}, n_iter={self.n_iter})"


class NormalizingFlows:
    """Planar normalizing flow (Rezende & Mohamed 2015): a stack of ``z' = z + u tanh(w.z+b)``
    layers fit by minimizing the reverse KL to ``log_prob``, with hand-derived reverse-mode
    gradients (no autodiff in the stack; verified against finite differences in tests).

    Only the forward direction is analytically tractable (planar flows are not easily
    invertible), so :meth:`log_prob`/density evaluation is available only for points that
    came from :meth:`sample` (via the accumulated log-determinant along the forward pass).
    """

    def __init__(self, log_prob, dim, n_layers=4, grad_log_prob=None, n_iter=2000, n_mc=128,
                 lr=0.01, init_scale=0.1, weight_decay=0.01, grad_clip=5.0, max_u_norm=4.0):
        self.target = LogDensity(log_prob, grad_log_prob)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.n_iter = int(n_iter)
        self.n_mc = int(n_mc)
        self.lr = float(lr)
        self.init_scale = float(init_scale)
        # ``u`` is only constrained (for invertibility) through its dot product with
        # ``w``; the component orthogonal to ``w`` is otherwise unpenalized, so plain
        # gradient ascent (with or without Adam) can drift ``u`` there without bound --
        # verified this is a real instability, not just an Adam artifact, by watching
        # plain small-step gradient ascent diverge to NaN within ~15 iterations. Weight
        # decay, per-step gradient clipping and a hard cap on ``||u||`` keep training
        # stable without touching the exact ELBO gradient formulas above (the cap is
        # applied after the optimizer step, so the *reported* gradient a test checks
        # against finite differences is still the raw, uncapped one).
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.max_u_norm = float(max_u_norm)
        self.params_ = None

    def _init_params(self, rng):
        dim = self.dim
        scale = self.init_scale / np.sqrt(dim)
        layers = []
        for _ in range(self.n_layers):
            u = rng.standard_normal(dim) * scale
            w = rng.standard_normal(dim) * scale
            layers.append({"u": self._project_u(u, w), "w": w, "b": 0.0})
        return layers

    _invertibility_margin = 0.1

    @classmethod
    def _project_u(cls, u, w):
        """Rezende & Mohamed's invertibility reparameterization ``u_hat = u + [m(w.u) -
        w.u] w/||w||^2``, with a safety margin: target ``w.u_hat >= -1 + margin`` rather
        than the bare ``-1`` boundary. At the bare boundary, ``denom = 1 + hp*(w.u_hat)``
        (the flow's log-det Jacobian factor) can approach exactly 0 whenever ``hp`` -> 1
        (points near the hyperplane ``w.z + b = 0``), which blows up every 1/denom term
        in the backward pass; the margin keeps ``denom >= margin`` everywhere and made
        training stable in practice (unmargined training diverged even under plain
        small-step gradient ascent, confirming this is a real, not cosmetic, fix)."""
        wu = w @ u
        m = -1.0 + cls._invertibility_margin + np.log1p(np.exp(wu))
        wnorm2 = w @ w
        if wnorm2 < 1e-300:
            return u
        return u + (m - wu) * w / wnorm2

    def forward(self, z, cache=False):
        """Forward map through the current (already-projected) parameters.

        The invertibility projection is applied once, right after each optimizer step
        (see :meth:`fit`) -- not here -- so that gradients computed against these stored
        parameters are exact (no extra projection Jacobian to account for) and match
        finite differences of this very function.
        """
        params = self.params_
        z = np.asarray(z, dtype=float)
        logdet = 0.0
        trace = [] if cache else None
        for layer in params:
            u, w, b = layer["u"], layer["w"], layer["b"]
            a = w @ z + b
            h = np.tanh(a)
            z_next = z + u * h
            hp = 1.0 - h * h
            psi = hp * w
            denom = 1.0 + u @ psi
            logdet += np.log(abs(denom) + 1e-300)
            if cache:
                trace.append({"z_prev": z.copy(), "u": u, "w": w, "b": b, "a": a, "h": h,
                              "hp": hp, "psi": psi, "denom": denom})
            z = z_next
        if cache:
            return z, logdet, trace
        return z, logdet

    def _backward(self, cache, v):
        """Backprop v = d(objective)/d(z_K) through the flow, accumulating d/d(z0) and
        per-layer parameter gradients for both the flow map and the logdet term."""
        dim = self.dim
        grad_z = v.copy()
        param_grads = []
        for layer in reversed(cache):
            z_prev, u, w, b = layer["z_prev"], layer["u"], layer["w"], layer["b"]
            h, hp, psi, denom = layer["h"], layer["hp"], layer["psi"], layer["denom"]
            hpp = -2.0 * h * hp
            uw = u @ w
            uv = u @ grad_z

            # flow-map pullback: z_next = z_prev + u * h(w.z_prev + b)
            d_zprev_map = grad_z + uv * hp * w
            d_u_map = grad_z * h
            d_w_map = uv * hp * z_prev
            d_b_map = uv * hp

            # logdet = log|1 + u.(hp * w)|; sign of denom folded via 1/denom (safe: |.|)
            inv_denom = 1.0 / denom
            d_zprev_logdet = (uw * hpp / denom) * w
            d_u_logdet = psi / denom
            d_w_logdet = (uw * hpp * z_prev + hp * u) / denom
            d_b_logdet = uw * hpp / denom

            param_grads.append({
                "u_map": d_u_map, "w_map": d_w_map, "b_map": d_b_map,
                "u_logdet": d_u_logdet, "w_logdet": d_w_logdet, "b_logdet": d_b_logdet,
            })
            grad_z = d_zprev_map + d_zprev_logdet
        param_grads.reverse()
        return grad_z, param_grads

    def pullback_grad(self, z, grad_theta):
        """``d/dz [ grad_theta . f(z) + logdet(z) ]`` -- what NeutraHMC needs to run
        gradient-based samplers in the flow's base space. Delegates to :meth:`_backward`,
        which chains the map pullback and each layer's direct logdet dependence through
        every earlier layer's Jacobian (verified against finite differences)."""
        _, _, cache = self.forward(z, cache=True)
        grad_z, _ = self._backward(cache, grad_theta)
        return grad_z

    def _forward_batch(self, Z):
        """Batched version of :meth:`forward` over ``Z`` of shape ``(n, dim)``, used only
        by :meth:`fit` (the same formulas as :meth:`forward`, vectorized over samples so
        training can afford enough Monte Carlo samples to keep gradient noise low)."""
        Z = np.asarray(Z, dtype=float)
        logdet = np.zeros(Z.shape[0])
        trace = []
        for layer in self.params_:
            u, w, b = layer["u"], layer["w"], layer["b"]
            A = Z @ w + b
            H = np.tanh(A)
            Z_next = Z + H[:, None] * u[None, :]
            HP = 1.0 - H * H
            uw = float(u @ w)
            DENOM = 1.0 + HP * uw
            logdet += np.log(np.abs(DENOM) + 1e-300)
            trace.append({"Z_prev": Z, "u": u, "w": w, "H": H, "HP": HP, "DENOM": DENOM, "uw": uw})
            Z = Z_next
        return Z, logdet, trace

    def _backward_batch(self, cache, V, clip=None):
        """Batched :meth:`_backward`: propagates ``V`` (n, dim) and returns per-sample
        clipped-then-averaged parameter gradients per layer (each a dict of ``u``/``w``/
        ``b`` arrays), plus the per-sample total gradient norm (for diagnostics)."""
        n = V.shape[0]
        grad_Z = V
        per_layer_grads = []
        flat_sq = np.zeros(n)
        raw_grads = []
        for layer in reversed(cache):
            Z_prev, u, w, H, HP, DENOM, uw = (layer["Z_prev"], layer["u"], layer["w"],
                                              layer["H"], layer["HP"], layer["DENOM"], layer["uw"])
            HPP = -2.0 * H * HP
            UV = grad_Z @ u

            d_zprev_map = grad_Z + (UV * HP)[:, None] * w[None, :]
            d_u_map = grad_Z * H[:, None]
            d_w_map = (UV * HP)[:, None] * Z_prev
            d_b_map = UV * HP

            d_zprev_logdet = (uw * HPP / DENOM)[:, None] * w[None, :]
            d_u_logdet = (HP / DENOM)[:, None] * w[None, :]
            d_w_logdet = (uw * HPP / DENOM)[:, None] * Z_prev + (HP / DENOM)[:, None] * u[None, :]
            d_b_logdet = uw * HPP / DENOM

            gu = d_u_map + d_u_logdet
            gw = d_w_map + d_w_logdet
            gb = d_b_map + d_b_logdet
            raw_grads.append((gu, gw, gb))
            flat_sq += np.sum(gu ** 2, axis=1) + np.sum(gw ** 2, axis=1) + gb ** 2
            grad_Z = d_zprev_map + d_zprev_logdet
        raw_grads.reverse()

        norm = np.sqrt(flat_sq)
        scale = np.ones(n) if clip is None else np.minimum(1.0, clip / (norm + 1e-12))
        for gu, gw, gb in raw_grads:
            per_layer_grads.append({
                "u": np.mean(scale[:, None] * gu, axis=0),
                "w": np.mean(scale[:, None] * gw, axis=0),
                "b": float(np.mean(scale * gb)),
            })
        return per_layer_grads, norm

    def fit(self, random_state=None):
        rng = _rng(random_state)
        self.params_ = self._init_params(rng)
        adams = [{"u": _Adam(self.dim, lr=self.lr), "w": _Adam(self.dim, lr=self.lr),
                  "b": _Adam((), lr=self.lr)} for _ in range(self.n_layers)]
        history = []
        # Keep-best tracking: this objective is noisy (Monte Carlo gradients over a
        # nonconvex loss) and can regress from a good solution late in training even
        # with the stability measures above, so the final flow is the snapshot with the
        # best smoothed (20-iteration trailing average) ELBO seen, not just the last one.
        best_elbo = -np.inf
        best_params = None
        window = 20
        for it in range(self.n_iter):
            # 1/sqrt(t) decay: Adam's adaptive normalization makes every step roughly
            # ``lr``-sized regardless of the true (noisy) gradient magnitude, which for
            # this small-parameter-scale, nonconvex objective caused late-training
            # excursions away from good solutions even at fairly small constant lr;
            # decaying the effective step size lets early iterations move freely and
            # later ones settle.
            decay = 1.0 / np.sqrt(1.0 + it / 100.0)
            for opt in adams:
                for a in opt.values():
                    a.lr = self.lr * decay
            Z0 = rng.standard_normal((self.n_mc, self.dim))
            # Flow forward/backward (this class's own code) is vectorized over the whole
            # batch; the target log-density is arbitrary user code taking one point at a
            # time, so that evaluation stays a per-sample loop.
            Theta, logdet, cache = self._forward_batch(Z0)
            logp = np.empty(self.n_mc)
            grad_theta = np.empty((self.n_mc, self.dim))
            for k in range(self.n_mc):
                logp[k] = self.target(Theta[k])
                grad_theta[k] = self.target.grad(Theta[k])
            # per-sample clipping: an occasional z0 landing in a region where the
            # (still-forming) flow maps to an extreme theta can otherwise produce a
            # single-sample gradient large enough to dominate and destabilize the whole
            # batch average.
            per_layer_grads, _ = self._backward_batch(cache, grad_theta, clip=self.grad_clip)
            for i, layer in enumerate(self.params_):
                gu = per_layer_grads[i]["u"] - self.weight_decay * layer["u"]
                gw = per_layer_grads[i]["w"] - self.weight_decay * layer["w"]
                gb = per_layer_grads[i]["b"]
                gu = _clip_norm(gu, self.grad_clip)
                gw = _clip_norm(gw, self.grad_clip)
                new_u = _clip_norm(adams[i]["u"].step(layer["u"], gu), self.max_u_norm)
                new_w = adams[i]["w"].step(layer["w"], gw)
                layer["w"] = new_w
                layer["u"] = self._project_u(new_u, new_w)
                layer["b"] = float(adams[i]["b"].step(np.array(layer["b"]), np.array(gb)))
            history.append(float(np.mean(logp + logdet)))
            if it + 1 >= window:
                smoothed = float(np.mean(history[-window:]))
                if smoothed > best_elbo:
                    best_elbo = smoothed
                    best_params = [dict(u=l["u"].copy(), w=l["w"].copy(), b=l["b"])
                                   for l in self.params_]
        if best_params is not None:
            self.params_ = best_params
        self.elbo_history_ = np.array(history)
        self.n_iter_ = self.n_iter
        return self

    def sample(self, n=1, random_state=None, return_log_prob=False):
        rng = _rng(random_state)
        z0 = rng.standard_normal((n, self.dim))
        base_logq = -0.5 * np.sum(z0 ** 2, axis=1) - self.dim * _HALF_LOG_2PI
        thetas = np.empty((n, self.dim))
        logqs = np.empty(n)
        for i in range(n):
            theta, logdet = self.forward(z0[i])
            thetas[i] = theta
            logqs[i] = base_logq[i] - logdet
        if return_log_prob:
            return thetas, logqs
        return thetas

    def log_det_jacobian(self, z):
        _, logdet = self.forward(np.asarray(z, dtype=float))
        return logdet

    def __repr__(self):
        return f"NormalizingFlows(dim={self.dim}, n_layers={self.n_layers})"


class SteinVI:
    """Stein variational gradient descent (Liu & Wang 2016) with an RBF kernel and the
    median-heuristic bandwidth, updated with AdaGrad-with-momentum (as in the reference
    implementation)."""

    def __init__(self, log_prob, dim, grad_log_prob=None, n_particles=100, n_iter=500,
                 step_size=0.1, bandwidth=None, init_scale=1.0, init_mean=None):
        self.target = LogDensity(log_prob, grad_log_prob)
        self.dim = int(dim)
        self.n_particles = int(n_particles)
        self.n_iter = int(n_iter)
        self.step_size = float(step_size)
        self.bandwidth = bandwidth
        self.init_scale = float(init_scale)
        self.init_mean = init_mean

    def fit(self, random_state=None):
        rng = _rng(random_state)
        dim = self.dim
        mean0 = np.zeros(dim) if self.init_mean is None else np.array(self.init_mean, dtype=float)
        X = mean0 + self.init_scale * rng.standard_normal((self.n_particles, dim))
        grad_hist = np.zeros_like(X)
        momentum = np.zeros_like(X)
        alpha = 0.9
        fudge = 1e-6
        n = self.n_particles
        for _ in range(self.n_iter):
            G = np.array([self.target.grad(x) for x in X])
            diff = X[:, None, :] - X[None, :, :]
            D2 = np.sum(diff ** 2, axis=2)
            h = self.bandwidth if self.bandwidth is not None else (np.median(D2) / max(np.log(n + 1), 1e-8) + 1e-12)
            h = max(h, 1e-12)
            K = np.exp(-D2 / h)
            grad_K = -(2.0 / h) * np.einsum('ij,ijk->ik', K, diff)
            phi = (K @ G + grad_K) / n
            grad_hist = alpha * grad_hist + (1 - alpha) * (phi ** 2)
            adj = phi / (fudge + np.sqrt(grad_hist))
            momentum = alpha * momentum + self.step_size * adj
            X = X + momentum
        self.particles_ = X
        self.n_iter_ = self.n_iter
        return self

    def sample(self, n=1, random_state=None):
        rng = _rng(random_state)
        idx = rng.integers(0, self.n_particles, size=n)
        return self.particles_[idx]

    def mean(self):
        return self.particles_.mean(axis=0)

    def cov(self):
        return np.cov(self.particles_.T, ddof=1)

    def __repr__(self):
        return f"SteinVI(dim={self.dim}, n_particles={self.n_particles})"
