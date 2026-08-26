"""``spl`` - the stochpylib command-line interface.

Installed as a console script via ``[project.scripts]`` in pyproject.toml.

Top-level flags:

- ``spl --help``            overview of what the installed library offers (modules, API
                            shape, quick-start snippet). The module inventory is
                            generated from each module's ``__all__`` so it never goes
                            stale.
- ``spl --version``         prints the installed version (pip metadata, falling back to
                            the in-code version) plus the latest version published on
                            PyPI (cached 24 h, non-blocking, offline-safe; disable all
                            PyPI traffic with STOCHPYLIB_SKIP_UPDATE_CHECK=1).
- ``spl --version --list``  additionally lists every version ever published on PyPI,
                            marking the installed one (``*``) and the latest (``latest``).
- ``spl --test``            runs the embedded self-check suite (see
                            :mod:`stochpylib.selftest`) against the installed copy -
                            works after any ``pip install``, no pytest or source
                            checkout needed.

Subcommands:

- ``spl update [--vers X] [--yes] [--dry-run] [--force]``
                            switch the installed PyPI package to any published
                            version (no ``--vers`` = latest). Validates the target,
                            shows the exact pip command, asks for confirmation unless
                            ``--yes``; ``--dry-run`` stops before executing; editable
                            installs are refused without ``--force``.
- ``spl info``              environment report: python/platform, numpy/scipy versions,
                            stochpylib version and install mode, module inventory.
- ``spl show <Name>``       signature + docstring of any public name (searches every
                            module; suggests close matches on a miss).
- ``spl demo [module]``     runs a live mini-example for a module (bare: lists demos).
                            See :mod:`stochpylib.cli_demo`.
- ``spl cite``              citation text (plain + BibTeX) for research use.
"""

import argparse
import difflib
import sys

from stochpylib.cli_pypi import (
    fetch_pypi_meta,
    install_mode,
    update_available,
)


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


def _module_public_names(module_name):
    """Public names exported by one installed module (via its __all__)."""
    import stochpylib

    mod = getattr(stochpylib, module_name)
    return list(getattr(mod, "__all__", []))


def _implemented_overview():
    """Human-readable inventory of what the installed package actually provides.

    Generated from each module's ``__all__`` (plus a hand-written one-line
    summary per module) so newly shipped modules can never be forgotten here.
    The distributions block additionally lists every class name, dynamically.
    """
    import textwrap

    import stochpylib

    summaries = {
        "probability":
            "sample spaces, events, P(), conditional_P(), bayes_theorem(),\n"
            "                   total_probability(), combinatorics (factorial ... derangements,\n"
            "                   Stirling, Bell, Catalan), independence/exclusion checks",
        "distributions":
            "47 distribution classes behind ONE common interface:\n"
            "                   .pdf()/.pmf(), .cdf(), .ppf(), .rvs(), .mean(), .var(),\n"
            "                   .skewness(), .kurtosis(), .entropy(), .mgf(), .cf(),\n"
            "                   .fit(), .ks_test()",
        "montecarlo":
            "quasi-random sequences (Sobol, Halton, Faure, Niederreiter,\n"
            "                   digital nets), crude_mc() / QMC / importance / rejection /\n"
            "                   stratified estimators with honest standard errors, variance\n"
            "                   reduction (antithetic, control variates, Latin hypercube,\n"
            "                   conditioned MC), applications (integration, pi, option\n"
            "                   pricing, VaR/ES, reliability, sensitivity)",
        "timeseries":
            "AR/MA/ARMA/ARIMA/SARIMA/ARFIMA, VAR/VARMA/VECM, GARCH family\n"
            "                   (ARCH..FIGARCH, MGARCH/DCC), Kalman/EKF/UKF/particle\n"
            "                   filters, HMM & regime switching, changepoint detection,\n"
            "                   spectral analysis (periodogram, CWT/DWT/STFT/Hilbert),\n"
            "                   ADF/KPSS/Ljung-Box/Granger diagnostics, forecasting &\n"
            "                   backtesting",
        "gaussian_processes":
            "GP regression & classification: composable kernel zoo\n"
            "                   (RBF, Matern, Periodic, Linear, Polynomial,\n"
            "                   RationalQuadratic, WhiteNoise, SpectralMixture, NN,\n"
            "                   ArcCosine) with +/*/** operators, exact inference,\n"
            "                   Laplace/EP/VI classification engines, FITC/VFE sparse\n"
            "                   approximations, DeepGP, hyperparameter optimization",
        "copulas":
            "dependence modeling: elliptical (Gaussian, Student-t),\n"
            "                   Archimedean (Clayton, Gumbel, Frank, Joe, AMH, BB1, BB7)\n"
            "                   + Plackett, empirical copulas (Empirical, Checkerboard,\n"
            "                   Beta/Bernstein), C-/D-/R-vines with AIC pair selection &\n"
            "                   rotations, CopulaFit dispatcher, Kendall's tau /\n"
            "                   Spearman's rho / tail dependence",
        "survival":
            "survival & reliability analysis: Kaplan-Meier, Nelson-Aalen,\n"
            "                   life tables, parametric fits (Weibull, Exponential,\n"
            "                   LogNormal, LogLogistic, Gompertz), Cox PH (Breslow/Efron)\n"
            "                   with stratification, Weibull AFT, Aalen additive hazards,\n"
            "                   Fine-Gray competing risks, log-rank test family,\n"
            "                   Aalen-Johansen CIF",
        "queueing":
            "queueing theory & networks: M/M/1, M/M/c, M/M/inf, M/D/1,\n"
            "                   M/G/1, GI/G/1 (Kingman), priority queues, Erlang B/C and\n"
            "                   Engset formulas, Jackson/closed/BCMP networks with\n"
            "                   mean-value analysis, Little's law, discrete-event\n"
            "                   simulation with warmup filtering",
        "information_theory":
            "information-theoretic measures: Shannon/Renyi/Tsallis/\n"
            "                   differential/max entropy, divergences (KL, Jensen-Shannon,\n"
            "                   Wasserstein, Hellinger, TV, chi-square, alpha), mutual-\n"
            "                   information quantities, channel capacity (BSC/BEC/Z),\n"
            "                   transfer entropy, Huffman coding, typical sets, AEP",
    }

    blocks = []
    total_names = 0
    for module_name in stochpylib.__all__:
        if not hasattr(stochpylib, module_name):
            continue
        names = _module_public_names(module_name)
        total_names += len(names)
        label = module_name
        pad = "\n                   " if len(label) > 17 else " " * (17 - len(label))
        summary = summaries.get(module_name, "see the module README")
        block = f"  {label}{pad}{summary}\n                   [{len(names)} public names]"
        if module_name == "distributions":
            class_names = [n for n in names
                           if n not in ("Distribution", "MultivariateDistribution")]
            name_block = textwrap.fill(
                ", ".join(sorted(class_names)),
                width=88,
                initial_indent="                   ",
                subsequent_indent="                   ",
            )
            block = f"  {label}{pad}{summary}\n{name_block}\n" \
                    f"                   [{len(names)} public names]"
        blocks.append(block)
    header = (f"  implemented modules: {len(stochpylib.__all__)} of 23 planned, "
              f"{total_names} public names total\n")
    return header + "\n".join(blocks)


def cmd_version(list_versions=False, _meta=None):
    """Print installed version + PyPI latest (and optionally all releases)."""
    installed = get_version()
    print(installed)
    meta = _meta if _meta is not None else fetch_pypi_meta()
    if meta is None or not meta.get("latest"):
        print("PyPI check unavailable (offline, or disabled via "
              "STOCHPYLIB_SKIP_UPDATE_CHECK).")
        meta = None
    else:
        status = update_available(installed, meta)
        latest = meta.get("latest", "")
        if status == "update":
            print(f"latest on PyPI: {latest}  (update available: run 'spl update')")
        elif status == "current":
            print(f"latest on PyPI: {latest}  (up to date)")
        else:
            print(f"latest on PyPI: {latest}  (installed version is newer / unreleased)")
    if list_versions:
        releases = (meta or {}).get("releases", [])
        if not releases:
            print("no published releases found on PyPI")
            return 0
        print(f"{len(releases)} published versions:")
        for rel in releases:
            marks = []
            if rel == installed:
                marks.append("* installed")
            if meta is not None and rel == meta.get("latest"):
                marks.append("latest")
            print(f"  {rel:<14}{'  '.join(marks)}")
    return 0


_UNSET = object()


def cmd_update(vers=None, yes=False, dry_run=False, force=False,
               _meta=_UNSET, _run=None, _input=None, _mode=None):
    """Switch the installed PyPI package to another published version.

    ``_meta`` is an injection point for tests: passing ``None`` simulates an
    unreachable PyPI; leaving it unset performs the real (forced) fetch.
    """
    installed = get_version()
    mode = _mode() if _mode is not None else install_mode()
    if mode in ("editable", "source", "local") and not force:
        print(f"stochpylib is installed from source ({mode}); 'spl update' manages")
        print("the pip-installed package and would not affect this checkout.")
        print("Re-install from source instead, or override with --force.")
        return 1
    meta = fetch_pypi_meta(force=True) if _meta is _UNSET else _meta
    if meta is None or not meta.get("releases"):
        print("Could not reach PyPI - nothing to compare against "
              "(check your connection, or set STOCHPYLIB_SKIP_UPDATE_CHECK=0).")
        return 1
    target = vers if vers is not None else meta["latest"]
    if target not in meta["releases"]:
        print(f"version {target} is not published on PyPI.")
        known = meta["releases"][-6:]
        print(f"recent published versions: {', '.join(known)}")
        return 1
    cmd = [sys.executable, "-m", "pip", "install", f"stochpylib=={target}"]
    print("Update plan:")
    print(f"  installed : {installed} ({mode})")
    print(f"  target    : {target}")
    print(f"  command   : {' '.join(cmd)}")
    if dry_run:
        print("Dry run - nothing executed.")
        return 0
    if not yes:
        ask = _input or input
        answer = ask("Proceed with the pip command above? [y/N] ")
        if str(answer).strip().lower() not in ("y", "yes"):
            print("Aborted - nothing changed.")
            return 1
    import subprocess

    runner = _run if _run is not None else subprocess.run
    result = runner(cmd)
    code = getattr(result, "returncode", 0)
    if code == 0:
        print(f"Done - stochpylib {target} installed. Re-run 'spl --version' to verify.")
    else:
        print(f"pip exited with code {code}; the installation was not changed by spl.")
    return code


def cmd_info():
    """Environment report: interpreter, dependencies, install mode, inventory."""
    import platform

    import numpy
    import scipy

    import stochpylib

    print(f"stochpylib {get_version()} ({install_mode()} install)")
    print(f"  python : {platform.python_version()} ({platform.system()} {platform.release()})")
    print(f"  numpy  : {numpy.__version__}")
    print(f"  scipy  : {scipy.__version__}")
    total = 0
    print("  modules:")
    for name in stochpylib.__all__:
        n = len(_module_public_names(name))
        total += n
        print(f"    {name:<20}{n:>4} public names")
    print(f"  total  : {total} public names across {len(stochpylib.__all__)} modules")
    print("  verify : spl --test (offline self-check suite, no pytest needed)")
    return 0


def cmd_show(name):
    """Print the qualified path, signature and docstring of a public name."""
    import inspect

    import stochpylib

    candidates = {}
    for module_name in stochpylib.__all__:
        mod = getattr(stochpylib, module_name, None)
        if mod is None:
            continue
        for attr in getattr(mod, "__all__", []):
            candidates.setdefault(attr, f"stochpylib.{module_name}")
    if name in candidates:
        module = __import__(candidates[name], fromlist=[name])
        obj = getattr(module, name)
        print(f"{candidates[name]}.{name}")
        try:
            print(inspect.signature(obj))
        except (TypeError, ValueError):
            pass  # builtins / classes without clean signatures
        doc = inspect.getdoc(obj)
        if doc:
            print()
            print(doc)
        return 0
    lower_map = {k.lower(): k for k in candidates}
    if name.lower() in lower_map:
        return cmd_show(lower_map[name.lower()])
    close = difflib.get_close_matches(name, list(candidates), n=5, cutoff=0.4)
    print(f"unknown public name: {name!r}")
    if close:
        print(f"did you mean: {', '.join(close)}?")
    return 1


def cmd_cite():
    """Print citation text (plain + BibTeX) for research use."""
    import stochpylib

    print("If stochpylib is useful in research, please cite it:")
    print()
    print("  Leon Schwarzkopf (2026). stochpylib: probability, distributions,")
    print("  stochastic processes, and statistical computing in one package.")
    print(f"  Version {get_version()}.")
    print("  https://github.com/leon1706-lol/Stochpylib")
    print()
    print("BibTeX:")
    print("@misc{stochpylib,")
    print("  author       = {Schwarzkopf, Leon},")
    print("  title        = {{stochpylib}: probability, distributions, stochastic")
    print("                 processes, and statistical computing in one package},")
    print("  year         = {2026},")
    print("  howpublished = {\\url{https://github.com/leon1706-lol/Stochpylib}},")
    print(f"  note         = {{Version {get_version()}}}")
    print("}")
    return 0


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

Try a live mini-example:  spl demo <module>   (bare 'spl demo' lists them)
"""

    parser = argparse.ArgumentParser(
        prog="spl",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "subcommands: update, info, show, demo, cite - run 'spl <command> --help'\n"
            "for details.\n\n"
            "roadmap: many more modules are planned (levy_processes,\n"
            "financial_stochastics, advanced_mcmc, bayesian, statistics, ...) -\n"
            "see the repository README and development/Implementation-Checklist.md.\n\n"
            "docs: README.md - contributing: CONTRIBUTING.md - security: SECURITY.md"
        ),
    )
    parser.add_argument(
        "--version", action="store_true",
        help="print the installed stochpylib version, then the latest version on PyPI")
    parser.add_argument(
        "--list", action="store_true", dest="list_versions",
        help="with --version: list every version published on PyPI "
             "(installed marked with '*', newest marked 'latest')")
    parser.add_argument(
        "--test", action="store_true", dest="run_selftest",
        help="run the built-in self-check suite against this installation")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    p_update = sub.add_parser(
        "update", help="switch the installed PyPI package to another version")
    p_update.add_argument("--vers", default=None, metavar="VERSION",
                          help="target version (default: latest published)")
    p_update.add_argument("--yes", action="store_true",
                          help="skip the confirmation prompt")
    p_update.add_argument("--dry-run", action="store_true", dest="dry_run",
                          help="print the plan and the exact pip command, execute nothing")
    p_update.add_argument("--force", action="store_true",
                          help="proceed even for editable/source installs")
    sub.add_parser("info", help="environment report: python/numpy/scipy versions, "
                                "install mode, module inventory")
    p_show = sub.add_parser("show", help="signature + docstring of a public name")
    p_show.add_argument("name", help="e.g. Normal, GARCH, bayes_theorem")
    p_demo = sub.add_parser("demo", help="run a live mini-example (bare: list demos)")
    p_demo.add_argument("module", nargs="?", default=None,
                        help="one of the implemented modules, e.g. queueing")
    sub.add_parser("cite", help="citation text (plain + BibTeX)")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(list_versions=args.list_versions)
    if args.run_selftest:
        from stochpylib.selftest import run

        failures = run(verbose=True)
        if failures:
            print(f"SELFTEST FAILED: {failures} failing check(s)", file=sys.stderr)
            return 1
        return 0
    if args.command == "update":
        return cmd_update(vers=args.vers, yes=args.yes, dry_run=args.dry_run,
                          force=args.force)
    if args.command == "info":
        return cmd_info()
    if args.command == "show":
        return cmd_show(args.name)
    if args.command == "demo":
        from stochpylib.cli_demo import run_demo

        return run_demo(args.module)
    if args.command == "cite":
        return cmd_cite()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
