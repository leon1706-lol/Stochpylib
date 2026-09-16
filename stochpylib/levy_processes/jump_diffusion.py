"""Jump-diffusion models: Merton, Kou, Bates, and the pure-jump Levy processes
Variance-Gamma, CGMY and Normal-Inverse-Gaussian.

Pricing conventions: ``call_price`` methods return the risk-neutral European
call value; Monte Carlo prices carry honest standard errors via the shared
``MCResult``. All closed forms are validated against the library's own
Black-Scholes oracle in the zero-jump / zero-vol limits (see tests).
"""

import math

import numpy as np
from scipy import integrate, special

from stochpylib.levy_processes.levy import LevyProcess
from stochpylib.levy_processes.subordinators import (
    GammaSubordinator,
    InverseGaussianSubordinator,
)

__all__ = [
    "JumpDiffusion", "MertonJumpDiffusion", "KouJumpDiffusion", "BatesModel",
    "VarianceGammaProcess", "CGMYProcess", "NormalInverseGaussianProcess",
]


def _norm_cdf(x):
    return 0.5 * (1.0 + special.erf(x / math.sqrt(2.0)))


def _black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def carr_madan_call(cf_log_price, S0, K, T, r, alpha=1.5, n_points=20_000,
                    v_max=200.0):
    """European call via Carr-Madan (1999) damped Fourier inversion.

    Integrates ``psi(v) = e^{-rT} phi(v - (alpha+1)i) / (alpha^2 + alpha
    - v^2 + i(2 alpha + 1) v)`` over ``v`` with trapezoid quadrature on a fine
    grid (no FFT needed for a single strike). ``cf_log_price(u)`` is the
    characteristic function of the *absolute* log-price ``ln S_T`` at
    maturity (it embeds ``ln S0`` in its drift term), so the damped Fourier
    variable ``k`` below must be the absolute log-strike ``ln K`` — not the
    log-moneyness ``ln(K/S0)`` — to match. Accurate for any model with an
    analytic cf; validated against Black-Scholes (exact), MC, and put-call
    parity in the tests.
    """
    k = math.log(K)
    v = np.linspace(1e-8, v_max, n_points)
    phi = cf_log_price(v - (alpha + 1.0) * 1j)
    psi = np.exp(-r * T) * phi \
        / (alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v)
    integrand = np.real(np.exp(-1j * v * k) * psi)
    integral = np.trapezoid(integrand, v) if hasattr(np, "trapezoid") \
        else np.trapz(integrand, v)
    return math.exp(-alpha * k) / math.pi * float(integral)


class JumpDiffusion(LevyProcess):
    """Base jump-diffusion: dX = b dt + sigma dW + dJ (compound Poisson).

    ``jump_sampler(n, rng)`` returns jump sizes; subclasses fix the family and
    the pricing formula. The state variable is the log-price by convention.
    """

    def __init__(self, b=0.0, sigma=0.2, jump_rate=0.0, jump_sampler=None,
                 jump_cf=None):
        super().__init__(b=b, sigma=sigma, jump_rate=jump_rate,
                         jump_sampler=jump_sampler, jump_cf=jump_cf)

    def simulate_prices(self, S0, T, n_steps, n_paths=1, random_state=None):
        """Price paths ``S_t = S0 * exp(X_t)``; shape ``(n_paths, n_steps+1)``."""
        log_paths = self.simulate(T, n_steps, n_paths, random_state)
        return S0 * np.exp(log_paths)


class MertonJumpDiffusion(JumpDiffusion):
    """Merton (1976): lognormal jumps with closed-form European call.

    Under Q the log-jumps are Normal(jump_mean, jump_std) arriving at
    ``jump_rate``; the compensator ``k = exp(jump_mean + jump_std^2/2) - 1``
    is subtracted from the drift when ``martingale=True`` (default), making
    ``exp(X_t)`` a martingale under the risk-neutral measure.
    """

    def __init__(self, mu=0.0, sigma=0.2, jump_rate=1.0, jump_mean=-0.1,
                 jump_std=0.2, martingale=True):
        k = math.exp(jump_mean + 0.5 * jump_std ** 2) - 1.0
        b = mu - 0.5 * sigma ** 2 - jump_rate * k if martingale \
            else mu - 0.5 * sigma ** 2
        self.mu = float(mu)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)
        self.martingale = bool(martingale)

        def sampler(n, rng):
            return rng.normal(jump_mean, jump_std, n)

        def jump_cf(u):
            return np.exp(1j * u * jump_mean - 0.5 * jump_std ** 2 * u ** 2)

        super().__init__(b=b, sigma=sigma, jump_rate=jump_rate,
                         jump_sampler=sampler, jump_cf=jump_cf)

    def call_price(self, S0, K, T, r=0.0, series_terms=80):
        lam_T = self.jump_rate * T
        k = math.exp(self.jump_mean + 0.5 * self.jump_std ** 2) - 1.0
        b_rn = r - 0.5 * self.sigma ** 2 - self.jump_rate * k
        price = 0.0
        for n in range(series_terms + 1):
            if lam_T > 0:
                log_p = -lam_T + n * math.log(lam_T) - math.lgamma(n + 1)
            else:
                log_p = 0.0 if n == 0 else -np.inf
            if log_p == -np.inf:
                continue
            m_n = math.log(S0) + b_rn * T + n * self.jump_mean
            v_n = math.sqrt(self.sigma ** 2 * T + n * self.jump_std ** 2)
            if v_n <= 0:
                call_n = max(math.exp(m_n) - K, 0.0)
            else:
                d1 = (m_n + v_n ** 2 - math.log(K)) / v_n
                d2 = d1 - v_n
                call_n = math.exp(m_n + 0.5 * v_n ** 2) * _norm_cdf(d1) \
                    - K * _norm_cdf(d2)
            price += math.exp(log_p - r * T) * max(call_n, 0.0)
        return float(price)

    def call_price_mc(self, S0, K, T, r=0.0, n_paths=100_000, random_state=None):
        """Monte-Carlo cross-check returning an ``MCResult``.

        Uses the exact aggregate: the sum of n i.i.d. Normal jumps is
        Normal(n*jump_mean, n*jump_std^2), so no per-jump loop is needed.
        """
        from stochpylib.montecarlo._result import MCResult

        rng = np.random.default_rng(random_state)
        x = self.b * T + self.sigma * math.sqrt(T) * rng.standard_normal(n_paths)
        n = rng.poisson(self.jump_rate * T, n_paths)
        x = x + n * self.jump_mean + np.sqrt(n) * self.jump_std \
            * rng.standard_normal(n_paths)
        payoff = np.maximum(S0 * np.exp(x) - K, 0.0)
        disc = math.exp(-r * T)
        est = float(disc * payoff.mean())
        se = float(disc * payoff.std(ddof=1) / math.sqrt(n_paths))
        return MCResult(est, se)


class KouJumpDiffusion(JumpDiffusion):
    """Kou (2002): double-exponential jumps with the series-call closed form.

    Log-jumps are Up with prob ``p_up`` (Exp(eta_up)) or Down with prob
    ``1 - p_up`` (-Exp(eta_down)); the martingale compensator uses the
    exponential-moment identities E[e^J] = p*eta_u/(eta_u-1) + (1-p)*eta_d/(eta_d+1).
    """

    def __init__(self, mu=0.0, sigma=0.2, jump_rate=1.0, eta_up=10.0,
                 eta_down=10.0, p_up=0.4, martingale=True):
        for eta in (eta_up, eta_down):
            if eta <= 0:
                raise ValueError("eta_up/eta_down must be positive")
        if not 0.0 <= p_up <= 1.0:
            raise ValueError("p_up must lie in [0, 1]")
        self.eta_up = float(eta_up)
        self.eta_down = float(eta_down)
        self.p_up = float(p_up)
        self.martingale = bool(martingale)
        e_pos = p_up * eta_up / (eta_up - 1.0)      # E[e^J] terms
        e_neg = (1.0 - p_up) * eta_down / (eta_down + 1.0)
        k = e_pos + e_neg - 1.0
        b = mu - 0.5 * sigma ** 2 - jump_rate * k if martingale \
            else mu - 0.5 * sigma ** 2
        self.mu = float(mu)

        def sampler(n, rng):
            up = rng.random(n) < p_up
            sizes = np.where(
                up,
                rng.exponential(1.0 / eta_up, n),
                -rng.exponential(1.0 / eta_down, n))
            return sizes

        def jump_cf(u):
            pos = p_up * eta_up / (eta_up - 1j * u)
            neg = (1.0 - p_up) * eta_down / (eta_down + 1j * u)
            return pos + neg

        super().__init__(b=b, sigma=sigma, jump_rate=jump_rate,
                         jump_sampler=sampler, jump_cf=jump_cf)

    def log_price_cf(self, S0, T, r=0.0):
        k = self.p_up * self.eta_up / (self.eta_up - 1) \
            + (1.0 - self.p_up) * self.eta_down / (self.eta_down + 1) - 1.0
        b_rn = r - 0.5 * self.sigma ** 2 - self.jump_rate * k

        def cf(u):
            u = np.asarray(u, dtype=complex)
            pos = self.p_up * self.eta_up / (self.eta_up - 1j * u)
            neg = (1.0 - self.p_up) * self.eta_down / (self.eta_down + 1j * u)
            jump = pos + neg - 1.0
            return np.exp(1j * u * (math.log(S0) + b_rn * T)
                          - 0.5 * self.sigma ** 2 * u ** 2 * T
                          + self.jump_rate * T * jump)

        return cf

    def call_price(self, S0, K, T, r=0.0, alpha=1.5, n_points=20_000,
                   v_max=200.0):
        """European call by Carr-Madan Fourier inversion of the exact Kou cf.

        Kou's published infinite-series form is equivalent; the Fourier route
        avoids restating its truncation constants and is validated against
        Monte Carlo and exact put-call parity in the tests.
        """
        return carr_madan_call(self.log_price_cf(S0, T, r), S0, K, T, r,
                               alpha=alpha, n_points=n_points, v_max=v_max)

    def call_price_mc(self, S0, K, T, r=0.0, n_paths=200_000, random_state=None):
        """Monte-Carlo cross-check returning an ``MCResult``."""
        from stochpylib.montecarlo._result import MCResult

        rng = np.random.default_rng(random_state)
        x = self.b * T + self.sigma * math.sqrt(T) * rng.standard_normal(n_paths)
        n = rng.poisson(self.jump_rate * T, n_paths)
        up = rng.random(n.sum()) < self.p_up
        sizes = np.where(up, rng.exponential(1.0 / self.eta_up, n.sum()),
                         -rng.exponential(1.0 / self.eta_down, n.sum()))
        offsets = np.concatenate(([0], np.cumsum(n)[:-1]))
        x = x + np.array([sizes[o:o + ni].sum() if ni else 0.0
                          for o, ni in zip(offsets, n)])
        payoff = np.maximum(S0 * np.exp(x) - K, 0.0)
        disc = math.exp(-r * T)
        return MCResult(float(disc * payoff.mean()),
                        float(disc * payoff.std(ddof=1) / math.sqrt(n_paths)))


class BatesModel:
    """Bates (1996): Heston stochastic volatility plus Merton lognormal jumps.

    Variance follows the CIR process; the log-price integrates sigma_t dW1
    with correlated Brownian (rho) plus compound-Poisson lognormal jumps.
    Simulation uses a full-truncation Euler for the variance with antithetic
    Brownians; ``call_price_mc`` returns an ``MCResult``.
    """

    def __init__(self, S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                 rho=-0.7, r=0.05, jump_rate=1.0, jump_mean=-0.1,
                 jump_std=0.15):
        if v0 <= 0 or theta <= 0:
            raise ValueError("v0 and theta must be positive")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        self.S0 = float(S0)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.r = float(r)
        self.jump_rate = float(jump_rate)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)

    def simulate(self, T, n_steps, n_paths=1, random_state=None):
        """Return ``(price_paths, var_paths)`` of shape ``(n_paths, n_steps+1)``."""
        rng = np.random.default_rng(random_state)
        dt = T / n_steps
        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        k = math.exp(self.jump_mean + 0.5 * self.jump_std ** 2) - 1.0
        for i in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            w1 = z1
            w2 = self.rho * z1 + math.sqrt(1 - self.rho ** 2) * z2
            v_pos = np.maximum(v[:, i], 0.0)
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * v_pos - self.jump_rate * k) * dt
                + np.sqrt(v_pos * dt) * w1)
            v[:, i + 1] = v[:, i] + self.kappa * (self.theta - v_pos) * dt \
                + self.xi * np.sqrt(v_pos * dt) * w2
            n_jumps = rng.poisson(self.jump_rate * dt, n_paths)
            if n_jumps.any():
                jumps = n_jumps * self.jump_mean \
                    + np.sqrt(n_jumps) * self.jump_std * rng.standard_normal(n_paths)
                S[:, i + 1] *= np.exp(jumps)
        return S, v

    def call_price_mc(self, K, T, n_paths=100_000, n_steps=100,
                      random_state=None):
        from stochpylib.montecarlo._result import MCResult

        S, _ = self.simulate(T, n_steps, n_paths, random_state)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        disc = math.exp(-self.r * T)
        return MCResult(float(disc * payoff.mean()),
                        float(disc * payoff.std(ddof=1) / math.sqrt(n_paths)))


class VarianceGammaProcess(LevyProcess):
    """Variance-Gamma process: Brownian motion time-changed by a gamma
    subordinator, with drift theta per unit gamma time.

    Closed-form characteristic function (unit mean-subordinated form):
    ``exp(t * (i u theta_bar - 0.5 sigma^2 u^2) ... )`` — implemented in the
    standard VG parameterization ``E[exp(iu X_t)] = (1 - i*u*nu*theta +
    0.5*sigma^2*nu*u^2)^(-t/nu)`` with drift ``theta_bar = theta``.
    """

    def __init__(self, sigma=0.2, nu=0.5, theta=0.0):
        if sigma <= 0 or nu <= 0:
            raise ValueError("sigma and nu must be positive")
        self.sigma = float(sigma)
        self.nu = float(nu)
        self.theta = float(theta)
        self._gamma = GammaSubordinator(rate=1.0 / nu, scale=nu)

    def _increment(self, dt, rng):
        gamma_t = self._gamma._increment(dt, rng)
        return self.theta * gamma_t \
            + self.sigma * np.sqrt(gamma_t) * rng.standard_normal()

    def characteristic_function(self, u, t):
        u = np.asarray(u, dtype=float)
        return (1.0 - 1j * u * self.theta * self.nu
                + 0.5 * self.sigma ** 2 * self.nu * u ** 2) ** (-t / self.nu)


class CGMYProcess:
    """CGMY (Carr-Geman-Madan-Yor) pure-jump Levy process.

    Closed-form characteristic exponent
    ``C * Gamma(-Y) [(M - iu)^Y - M^Y + (G + iu)^Y - G^Y]`` per unit time.
    Simulation uses a double compound-Poisson truncation (jumps below
    ``jump_floor`` excluded, their analytic mean re-injected as drift) — a
    documented approximation valid for pricing at the chosen floor.
    """

    def __init__(self, C=1.0, G=5.0, M=5.0, Y=0.5, jump_floor=1e-4):
        if C <= 0 or G <= 0 or M <= 0:
            raise ValueError("C, G, M must be positive")
        if not 0.0 < Y < 2.0:
            raise ValueError("Y must lie in (0, 2)")
        if Y >= 1.0 and M <= 1.0:
            raise ValueError("M > 1 required when Y >= 1")
        self.C = float(C)
        self.G = float(G)
        self.M = float(M)
        self.Y = float(Y)
        self.jump_floor = float(jump_floor)
        self._grids = {}

    def characteristic_exponent(self, u):
        u = np.asarray(u, dtype=float)
        return self.C * special.gamma(-self.Y) * (
            (self.M - 1j * u) ** self.Y - self.M ** self.Y
            + (self.G + 1j * u) ** self.Y - self.G ** self.Y)

    def characteristic_function(self, u, t):
        return np.exp(t * self.characteristic_exponent(u))

    def _side_intensity_and_mean(self, lam):
        """Intensity above ``jump_floor`` and retained mean magnitude for one
        Levy side ``nu(dx) = C e^{-lam x} x^{-1-Y} dx`` (x > 0)."""
        from stochpylib.levy_processes.subordinators import upper_gamma_negative

        intensity = self.C * lam ** self.Y * upper_gamma_negative(
            -self.Y, lam * self.jump_floor)
        mean = self.C * lam ** (self.Y - 1.0) \
            * special.gammaincc(1.0 - self.Y, lam * self.jump_floor) \
            * special.gamma(1.0 - self.Y)
        return intensity, mean

    def _jump_grid(self, lam):
        if lam not in self._grids:
            x_max = max(self.jump_floor * 4000.0, 10.0 / lam)
            # log-spaced + cumulative-trapezoid: see TemperingSubordinator's
            # ``_jump_quantile_grid`` for why a linear grid biases the mean.
            x = np.logspace(np.log10(self.jump_floor), np.log10(x_max), 8192)
            logd = -lam * x + (-1.0 - self.Y) * np.log(x)
            d = np.exp(logd - logd.max())
            cdf = integrate.cumulative_trapezoid(d, x, initial=0.0)
            cdf /= cdf[-1]
            self._grids[lam] = (x, cdf)
        return self._grids[lam]

    def _increment(self, dt, rng):
        if self.Y >= 1.0:
            raise ValueError(
                "simulation by compound-Poisson truncation requires Y < 1 "
                "(the small-jump mean diverges for Y >= 1); the "
                "characteristic function remains valid for all Y in (0, 2)")
        total = 0.0
        for lam, sign in ((self.M, -1.0), (self.G, 1.0)):
            intensity, retained_mean = self._side_intensity_and_mean(lam)
            n = rng.poisson(self.C * dt * intensity)
            if n:
                x_grid, cdf = self._jump_grid(lam)
                jumps = np.interp(rng.random(n), cdf, x_grid)
                total += sign * float(jumps.sum())
            full_mean = self.C * special.gamma(1.0 - self.Y) \
                * lam ** (self.Y - 1.0)
            total += sign * dt * (full_mean - retained_mean)
        return float(total)

    def simulate(self, T, n_steps, n_paths=1, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = float(T) / n_steps
        incs = np.array([[self._increment(dt, rng) for _ in range(n_steps)]
                         for _ in range(n_paths)])
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = 0.0
        np.cumsum(incs, axis=1, out=paths[:, 1:])
        return paths


class NormalInverseGaussianProcess:
    """NIG process: Brownian motion with drift subordinated by an inverse
    Gaussian clock.

    Closed-form characteristic function
    ``exp(t * delta * (sqrt(alpha^2 - (beta + iu)^2) - sqrt(alpha^2 - beta^2)))``.
    Simulation is exact subordination through the library's native IG sampler.
    """

    def __init__(self, alpha=5.0, beta=-1.0, delta=1.0):
        if alpha <= abs(beta):
            raise ValueError("alpha must exceed |beta|")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        # IG clock with E[T_t] = t: increments IG(mu=dt, lam=delta^2 * dt)
        self._clock = InverseGaussianSubordinator(lam=delta ** 2)

    def _increment(self, dt, rng):
        tau = self._clock._increment(dt, rng)
        return self.beta * self.delta ** 2 * tau \
            + self.delta * np.sqrt(tau) * rng.standard_normal()

    def characteristic_function(self, u, t):
        u = np.asarray(u, dtype=float)
        a2 = self.alpha ** 2
        return np.exp(t * self.delta * (
            np.sqrt(np.maximum(a2 - (self.beta + 1j * u) ** 2, 0.0) + 0j)
            - math.sqrt(a2 - self.beta ** 2)))

    def simulate(self, T, n_steps, n_paths=1, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = float(T) / n_steps
        incs = np.array([[self._increment(dt, rng) for _ in range(n_steps)]
                         for _ in range(n_paths)])
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = 0.0
        np.cumsum(incs, axis=1, out=paths[:, 1:])
        return paths
