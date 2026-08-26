"""Tests for the spl CLI expansion (V0.6.4): PyPI awareness, spl update,
spl info, spl show, spl demo, spl cite.

Every test is offline: PyPI responses are mocked (never fetched), pip is never
executed (subprocess mocked), and prompts are simulated. The live network path
is exercised only by the manual debugging session, not by CI.
"""

import json

import pytest

import stochpylib
import stochpylib.cli as cli
from stochpylib.cli_pypi import (
    install_mode,
    update_available,
    version_key,
)

PYPI_META = {
    "latest": "0.7.0",
    "releases": ["0.1.0", "0.1.1", "0.6.3", "0.7.0"],
}


# --------------------------------------------------------------- version_key

@pytest.mark.parametrize("version,expected", [
    ("0.6.3", (0, 6, 3)),
    ("1.0", (1, 0)),
    ("0.10.2", (0, 10, 2)),
    ("0.6.3rc1", (0, 6, 3)),
    ("", (0,)),
])
def test_version_key_parses(version, expected):
    assert version_key(version) == expected


def test_version_key_numeric_not_lexicographic():
    assert version_key("0.10.0") > version_key("0.9.9")


# ------------------------------------------------------------------- fetch

class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _pypi_payload():
    return {
        "info": {"version": "0.7.0"},
        "releases": {"0.1.0": [], "0.1.1": [], "0.6.3": [], "0.7.0": []},
    }


def test_fetch_parses_pypi_json(tmp_path, monkeypatch):
    import urllib.request as ur

    def fake_urlopen(req, timeout):
        return _FakeResponse(_pypi_payload())

    monkeypatch.setattr(ur, "urlopen", fake_urlopen)
    meta = cli.fetch_pypi_meta(cache_path=tmp_path / "c.json")
    assert meta["latest"] == "0.7.0"
    assert meta["releases"] == ["0.1.0", "0.1.1", "0.6.3", "0.7.0"]


def test_fetch_serves_fresh_cache_without_network(tmp_path, monkeypatch):
    import time

    import urllib.request as ur

    cache = tmp_path / "c.json"
    cache.write_text(json.dumps(
        {"timestamp": time.time(), "meta": PYPI_META}), encoding="utf-8")

    def boom(*a, **kw):
        raise AssertionError("network must not be touched with a fresh cache")

    monkeypatch.setattr(ur, "urlopen", boom)
    assert cli.fetch_pypi_meta(cache_path=cache) == PYPI_META


def test_fetch_expired_cache_refetches(tmp_path, monkeypatch):
    import time

    import urllib.request as ur

    cache = tmp_path / "c.json"
    cache.write_text(json.dumps(
        {"timestamp": time.time() - 10**9, "meta": {"latest": "0.1.0",
                                                    "releases": ["0.1.0"]}}),
        encoding="utf-8")
    monkeypatch.setattr(ur, "urlopen", lambda req, timeout: _FakeResponse(
        _pypi_payload()))
    meta = cli.fetch_pypi_meta(cache_path=cache)
    assert meta["latest"] == "0.7.0"


def test_fetch_offline_returns_none(tmp_path, monkeypatch):
    import urllib.request as ur

    def boom(req, timeout):
        raise OSError("no network")

    monkeypatch.setattr(ur, "urlopen", boom)
    assert cli.fetch_pypi_meta(cache_path=tmp_path / "c.json") is None


def test_fetch_skip_env_disables_everything(tmp_path, monkeypatch):
    import urllib.request as ur

    monkeypatch.setenv("STOCHPYLIB_SKIP_UPDATE_CHECK", "1")

    def boom(*a, **kw):
        raise AssertionError("skip env must prevent all network traffic")

    monkeypatch.setattr(ur, "urlopen", boom)
    assert cli.fetch_pypi_meta(cache_path=tmp_path / "c.json") is None


def test_fetch_unwritable_cache_still_returns_meta(tmp_path, monkeypatch):
    import urllib.request as ur

    monkeypatch.setattr(ur, "urlopen", lambda req, timeout: _FakeResponse(
        _pypi_payload()))
    unwritable = tmp_path / "no-such-dir" / "c.json"
    meta = cli.fetch_pypi_meta(cache_path=unwritable)
    assert meta is not None and meta["latest"] == "0.7.0"


# ------------------------------------------------------------ update_available

def test_update_available_statuses():
    assert update_available("0.6.3", {"latest": "0.7.0"}) == "update"
    assert update_available("0.7.0", {"latest": "0.7.0"}) == "current"
    assert update_available("0.8.0", {"latest": "0.7.0"}) == "newer"
    assert update_available("0.6.3", None) == "unknown"
    assert update_available("0.6.3", {"latest": ""}) == "unknown"


# ------------------------------------------------------------------ install_mode

def test_install_mode_returns_known_classification():
    assert install_mode() in ("editable", "local", "wheel", "source")


# --------------------------------------------------------------- spl --version

def test_version_cmd_reports_installed_and_latest(capsys, monkeypatch):
    monkeypatch.setattr(cli, "fetch_pypi_meta", lambda **kw: dict(PYPI_META))
    monkeypatch.setattr(cli, "get_version", lambda: "0.6.3")
    assert cli.main(["--version"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "0.6.3"
    assert "latest on PyPI: 0.7.0" in lines[1]
    assert "update available" in lines[1]


def test_version_cmd_up_to_date(capsys, monkeypatch):
    monkeypatch.setattr(cli, "fetch_pypi_meta", lambda **kw: dict(PYPI_META))
    monkeypatch.setattr(cli, "get_version", lambda: "0.7.0")
    assert cli.main(["--version"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_version_cmd_newer_than_pypi(capsys, monkeypatch):
    monkeypatch.setattr(cli, "fetch_pypi_meta", lambda **kw: dict(PYPI_META))
    monkeypatch.setattr(cli, "get_version", lambda: "0.8.0")
    assert cli.main(["--version"]) == 0
    assert "newer / unreleased" in capsys.readouterr().out


def test_version_cmd_offline_message(capsys, monkeypatch):
    monkeypatch.setattr(cli, "fetch_pypi_meta", lambda **kw: None)
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == cli.get_version()
    assert "PyPI check unavailable" in out[1]


def test_version_list_marks_installed_and_latest(capsys, monkeypatch):
    monkeypatch.setattr(cli, "fetch_pypi_meta", lambda **kw: dict(PYPI_META))
    monkeypatch.setattr(cli, "get_version", lambda: "0.6.3")
    assert cli.main(["--version", "--list"]) == 0
    out = capsys.readouterr().out
    assert "* installed" in out
    assert "latest" in out
    assert "4 published versions" in out
    # the installed release line (indented table row) carries the marker;
    # exclude the bare version echo on line 1
    installed_line = [ln for ln in out.splitlines()
                      if ln.startswith("  ") and ln.strip().startswith("0.6.3")][0]
    assert "* installed" in installed_line
    latest_line = [ln for ln in out.splitlines()
                   if ln.startswith("  ") and ln.strip().startswith("0.7.0")][0]
    assert "latest" in latest_line and "* installed" not in latest_line


def test_version_list_no_releases(capsys, monkeypatch):
    monkeypatch.setattr(cli, "fetch_pypi_meta",
                        lambda **kw: {"latest": "", "releases": []})
    monkeypatch.setattr(cli, "get_version", lambda: "0.6.3")
    assert cli.main(["--version", "--list"]) == 0
    assert "no published releases" in capsys.readouterr().out


# ------------------------------------------------------------------ spl update

def test_update_dry_run_prints_plan_without_executing(capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "fetch_pypi_meta", lambda **kw: dict(PYPI_META))
    monkeypatch.setattr(cli, "install_mode", lambda: "wheel")
    rc = cli.cmd_update(_meta=dict(PYPI_META), dry_run=True,
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "wheel")
    out = capsys.readouterr().out
    assert rc == 0
    assert ran == []
    assert "Dry run" in out
    assert "stochpylib==0.7.0" in out


def test_update_specific_version_builds_pip_command(capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "install_mode", lambda: "wheel")
    rc = cli.cmd_update(vers="0.6.3", yes=True, _meta=dict(PYPI_META),
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "wheel")
    out = capsys.readouterr().out
    assert rc == 0
    assert ran == [[__import__("sys").executable, "-m", "pip", "install",
                    "stochpylib==0.6.3"]]
    assert "target    : 0.6.3" in out


def test_update_defaults_to_latest(capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "install_mode", lambda: "wheel")
    rc = cli.cmd_update(yes=True, _meta=dict(PYPI_META),
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "wheel")
    assert rc == 0
    assert ran and ran[0][-1] == "stochpylib==0.7.0"


def test_update_rejects_unknown_version(capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "install_mode", lambda: "wheel")
    rc = cli.cmd_update(vers="9.9.9", yes=True, _meta=dict(PYPI_META),
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "wheel")
    out = capsys.readouterr().out
    assert rc == 1
    assert ran == []
    assert "9.9.9 is not published" in out
    assert "recent published versions" in out


def test_update_refuses_source_install_without_force(capsys):
    ran = []
    rc = cli.cmd_update(yes=True, _meta=dict(PYPI_META),
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "editable")
    out = capsys.readouterr().out
    assert rc == 1
    assert ran == []
    assert "installed from source" in out
    assert "--force" in out


def test_update_source_install_with_force_proceeds(capsys):
    ran = []
    rc = cli.cmd_update(yes=True, force=True, _meta=dict(PYPI_META),
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "editable")
    assert rc == 0
    assert len(ran) == 1


def test_update_prompt_aborts_on_n(capsys):
    ran = []
    rc = cli.cmd_update(_meta=dict(PYPI_META), _input=lambda _prompt="": "n",
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "wheel")
    out = capsys.readouterr().out
    assert rc == 1
    assert ran == []
    assert "Aborted" in out


def test_update_prompt_proceeds_on_y(capsys):
    ran = []
    rc = cli.cmd_update(_meta=dict(PYPI_META), _input=lambda _prompt="": "y",
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "wheel")
    assert rc == 0
    assert len(ran) == 1


def test_update_offline_fails_cleanly(capsys):
    ran = []
    rc = cli.cmd_update(yes=True, _meta=None,
                        _run=lambda cmd: ran.append(cmd) or _FakeResult(0),
                        _mode=lambda: "wheel")
    out = capsys.readouterr().out
    assert rc == 1
    assert ran == []
    assert "Could not reach PyPI" in out


def test_update_pip_failure_propagates_exit_code(capsys):
    rc = cli.cmd_update(yes=True, _meta=dict(PYPI_META),
                        _run=lambda cmd: _FakeResult(3),
                        _mode=lambda: "wheel")
    assert rc == 3


class _FakeResult:
    def __init__(self, returncode):
        self.returncode = returncode


# -------------------------------------------------------------------- spl info

def test_info_reports_environment(capsys):
    import numpy
    import scipy

    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert f"stochpylib {stochpylib.__version__}" in out
    assert numpy.__version__ in out
    assert scipy.__version__ in out
    for module in stochpylib.__all__:
        assert module in out


# -------------------------------------------------------------------- spl show

def test_show_known_class_prints_signature_and_doc(capsys):
    assert cli.main(["show", "Normal"]) == 0
    out = capsys.readouterr().out
    assert "stochpylib.distributions.Normal" in out
    assert "(mu" in out  # constructor signature
    assert len(out) > 100  # docstring present


def test_show_known_function(capsys):
    assert cli.main(["show", "bayes_theorem"]) == 0
    out = capsys.readouterr().out
    assert "stochpylib.probability.bayes_theorem" in out


def test_show_case_insensitive_fallback(capsys):
    assert cli.main(["show", "normal"]) == 0
    assert "stochpylib.distributions.Normal" in capsys.readouterr().out


def test_show_unknown_suggests_close_matches(capsys):
    assert cli.main(["show", "Nrmal"]) == 1
    out = capsys.readouterr().out
    assert "unknown public name" in out
    assert "Normal" in out  # suggestion


def test_show_every_exported_name_resolves():
    """spl show must be able to find every public name of every module."""
    for module_name in stochpylib.__all__:
        mod = getattr(stochpylib, module_name)
        for attr in getattr(mod, "__all__", []):
            assert cli.cmd_show(attr) == 0, f"{module_name}.{attr} not resolvable"


# -------------------------------------------------------------------- spl demo

def test_demo_bare_lists_all_modules(capsys):
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    for module in stochpylib.__all__:
        assert module in out


@pytest.mark.parametrize("module", sorted(stochpylib.__all__))
def test_demo_module_runs(module, capsys):
    assert cli.main(["demo", module]) == 0
    assert len(capsys.readouterr().out.strip()) > 40


def test_demo_unknown_module_fails_with_suggestion(capsys):
    assert cli.main(["demo", "probabilty"]) == 1
    out = capsys.readouterr().out
    assert "unknown demo module" in out
    assert "probability" in out


# -------------------------------------------------------------------- spl cite

def test_cite_contains_bibtex_and_urls(capsys):
    assert cli.main(["cite"]) == 0
    out = capsys.readouterr().out
    assert "@misc{stochpylib," in out
    assert "Schwarzkopf" in out
    assert "https://github.com/leon1706-lol/Stochpylib" in out
    assert stochpylib.__version__ in out


# ------------------------------------------------------------------- --help

def test_help_lists_subcommands_and_flags(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for token in ("--version", "--test", "--list", "update", "info",
                  "show", "demo", "cite"):
        assert token in out, f"spl --help lacks {token!r}"


def test_help_inventory_covers_all_modules_with_counts(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for module in stochpylib.__all__:
        assert module in out
    total = sum(len(getattr(getattr(stochpylib, m), "__all__", []))
                for m in stochpylib.__all__)
    assert f"{total} public names total" in out
