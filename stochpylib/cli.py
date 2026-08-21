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
            "roadmap: many more modules are planned (timeseries, gaussian_processes, copulas,\n"
            "survival, queueing, financial_stochastics, montecarlo, bayesian, statistics, ...) -\n"
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
