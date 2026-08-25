"""``spl`` - the stochpylib command-line interface.

Installed as a console script via ``[project.scripts]`` in pyproject.toml:

- ``spl --help``    overview of what the installed library offers (modules, API shape,
                    quick-start snippet)
- ``spl --version`` prints the installed stochpylib version (from package metadata,
                    falling back to the in-code version when not pip-installed).
- ``spl --test``    runs the embedded self-check suite (see :mod:`stochpylib.selftest`)
                    against the installed copy - works after any ``pip install``, no pytest
                    or source checkout needed.
"""

import argparse
import sys
import textwrap


def get_version():
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("stochpylib")
        except PackageNotFoundError:
            pass
    except ImportError:  # pragma: no cover (package requires >= 3.10 anyway)
        pass
    import stochpylib

    return stochpylib.__version__


def _implemented_overview():
    """Human-readable inventory of what the installed package actually provides."""
    import stochpylib

    blocks = []

    if hasattr(stochpylib, "probability"):
        blocks.append(
            "  probability      core probability engine: sample spaces, events, P(),\n"
            "                   conditional_P(), bayes_theorem(), total_probability(),\n"
            "                   combinatorics (factorial ... derangements, Stirling,\n"
            "                   Bell, Catalan), independence/exclusion checks"
        )

    if hasattr(stochpylib, "distributions"):
        import stochpylib.distributions as dists

        names = [n for n in dists.__all__ if n not in ("Distribution", "MultivariateDistribution")]
        name_block = textwrap.fill(
            ", ".join(sorted(names)),
            width=88,
            initial_indent="                   ",
            subsequent_indent="                   ",
        )
        blocks.append(
            f"  distributions    {len(names)} distribution classes behind ONE common interface:\n"
            "                   .pdf()/.pmf(), .cdf(), .ppf(), .rvs(), .mean(), .var(),\n"
            "                   .skewness(), .kurtosis(), .entropy(), .mgf(), .cf(),\n"
            "                   .fit(), .ks_test()\n"
            f"{name_block}"
        )

    if hasattr(stochpylib, "montecarlo"):
        blocks.append(
            "  montecarlo       quasi-random sequences (Sobol, Halton, Faure,\n"
            "                   Niederreiter, digital nets), crude_mc() / QMC /\n"
            "                   importance / rejection / stratified estimators with\n"
            "                   honest standard errors, variance reduction (antithetic,\n"
            "                   control variates, Latin hypercube, conditioned MC),\n"
            "                   applications (integration, pi, option pricing, VaR/ES,\n"
            "                   reliability, sensitivity)"
        )

    if hasattr(stochpylib, "timeseries"):
        blocks.append(
            "  timeseries       AR/MA/ARMA/ARIMA/SARIMA/ARFIMA, VAR/VARMA/VECM,\n"
            "                   GARCH family (ARCH..FIGARCH, MGARCH/DCC), Kalman/\n"
            "                   EKF/UKF/particle filters, HMM & regime switching,\n"
            "                   changepoint detection, spectral analysis (periodogram,\n"
            "                   CWT/DWT/STFT/Hilbert), ADF/KPSS/Ljung-Box/Granger\n"
            "                   diagnostics, forecasting & backtesting"
        )

    if hasattr(stochpylib, "gaussian_processes"):
        blocks.append(
            "  gaussian_processes\n"
            "                   GP regression & classification: composable kernel zoo\n"
            "                   (RBF, Matern, Periodic, Linear, Polynomial,\n"
            "                   RationalQuadratic, WhiteNoise, SpectralMixture, NN,\n"
            "                   ArcCosine) with +/*/** operators, exact inference,\n"
            "                   Laplace/EP/VI classification engines, FITC/VFE sparse\n"
            "                   approximations, DeepGP, hyperparameter optimization"
        )

    if hasattr(stochpylib, "copulas"):
        blocks.append(
            "  copulas          dependence modeling: elliptical (Gaussian, Student-t),\n"
            "                   Archimedean (Clayton, Gumbel, Frank, Joe, AMH, BB1,\n"
            "                   BB7) + Plackett, empirical copulas (Empirical,\n"
            "                   Checkerboard, Beta/Bernstein), C-/D-/R-vines with\n"
            "                   AIC pair selection & rotations, CopulaFit dispatcher,\n"
            "                   Kendall's tau / Spearman's rho / tail dependence"
        )

    if hasattr(stochpylib, "survival"):
        blocks.append(
            "  survival         survival & reliability analysis: Kaplan-Meier,\n"
            "                   Nelson-Aalen, life tables, parametric fits (Weibull,\n"
            "                   Exponential, LogNormal, LogLogistic, Gompertz),\n"
            "                   Cox PH (Breslow/Efron) with stratification, Weibull\n"
            "                   AFT, Aalen additive hazards, Fine-Gray competing\n"
            "                   risks, log-rank test family, Aalen-Johansen CIF"
        )

    return "\n".join(blocks)


def _build_parser():
    version = get_version()
    description = f"""\
stochpylib {version} - probability, distributions, stochastic processes, and statistical
computing in one package (aiming to replace stitching together scipy.stats, statsmodels,
pymc, arch, lifelines, copulas).

What the installed copy offers:

{_implemented_overview()}

Quick start:

    from stochpylib.probability import bayes_theorem, total_probability
    p_pos = total_probability((0.99, 0.01), (0.05, 0.99))
    p_disease = bayes_theorem(0.01, 0.99, p_pos)

    from stochpylib.distributions import Normal
    d = Normal(0.0, 1.0)
    d.pdf(0.0); d.cdf(1.96); d.ppf(0.975); d.rvs(100, random_state=0)
    fitted = Normal.fit(data)

Run everything from any Python prompt after `pip install stochpylib`.
"""

    parser = argparse.ArgumentParser(
        prog="spl",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "roadmap: many more modules are planned (levy_processes,\n"
            "financial_stochastics, advanced_mcmc, bayesian, statistics, ...) -\n"
            "see the repository README and development/Implementation-Checklist.md.\n\n"
            "docs: README.md - contributing: CONTRIBUTING.md - security: SECURITY.md"
        ),
    )
    parser.add_argument(
        "--version", action="store_true", help="print the installed stochpylib version and exit"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        dest="run_selftest",
        help="run the built-in self-check suite against this installation",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(get_version())
        return 0
    if args.run_selftest:
        from stochpylib.selftest import run

        failures = run(verbose=True)
        if failures:
            print(f"SELFTEST FAILED: {failures} failing check(s)", file=sys.stderr)
            return 1
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
