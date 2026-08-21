"""``spl`` — the stochpylib command-line interface.

Installed as a console script via ``[project.scripts]`` in pyproject.toml:

- ``spl --version`` prints the installed stochpylib version (from package metadata,
  falling back to the in-code version when not pip-installed).
- ``spl --test`` runs the embedded self-check suite (see :mod:`stochpylib.selftest`)
  against the installed copy — works after any ``pip install``, no pytest or source
  checkout needed.
"""

import argparse
import sys


def get_version():
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("stochpylib")
        except PackageNotFoundError:
            pass
    except ImportError:  # Python < 3.8 fallback (package requires >= 3.10 anyway)
        pass
    import stochpylib

    return stochpylib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="spl",
        description="stochpylib command line: inspect the installed library and verify it.",
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
