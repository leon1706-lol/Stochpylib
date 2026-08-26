"""PyPI metadata access for the ``spl`` CLI's version awareness.

Everything here is offline-safe by design: network failures, timeouts and a
disabled check all degrade to ``None`` / a clear message — never an exception
escaping into the CLI. The PyPI check result is cached on disk for 24 hours so
routine ``spl --version`` calls stay instant and polite to PyPI; ``--list`` and
``spl update`` always fetch fresh. Set ``STOCHPYLIB_SKIP_UPDATE_CHECK=1`` to
disable all PyPI traffic.
"""

import json
import os
import tempfile
import time
from pathlib import Path

__all__ = [
    "PYPI_JSON_URL", "CACHE_TTL_SECONDS", "FETCH_TIMEOUT_SECONDS",
    "SKIP_CHECK_ENV", "version_key", "fetch_pypi_meta",
    "install_mode", "update_available",
]

PYPI_JSON_URL = "https://pypi.org/pypi/stochpylib/json"
CACHE_TTL_SECONDS = 24 * 3600
FETCH_TIMEOUT_SECONDS = 4.0
SKIP_CHECK_ENV = "STOCHPYLIB_SKIP_UPDATE_CHECK"


def version_key(version):
    """Parse a version string into a comparable tuple of ints.

    ``"0.10.2" -> (0, 10, 2)`` so numeric ordering is preserved (string
    comparison would rank ``0.10`` below ``0.9``). Non-numeric suffixes
    (``0.7.0rc1``) are tolerated: the numeric prefix is used, missing
    components count as 0.
    """
    parts = []
    for piece in str(version).strip().split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts) if parts else (0,)


def _default_cache_path():
    return Path(tempfile.gettempdir()) / "stochpylib_pypi_meta.json"


def _read_cache(cache_path):
    try:
        raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if time.time() - float(raw["timestamp"]) <= CACHE_TTL_SECONDS:
            return raw["meta"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _write_cache(cache_path, meta):
    try:
        Path(cache_path).write_text(
            json.dumps({"timestamp": time.time(), "meta": meta}),
            encoding="utf-8")
    except OSError:
        pass  # a full/unwritable temp dir must never break the CLI


def fetch_pypi_meta(force=False, cache_path=None,
                    timeout=FETCH_TIMEOUT_SECONDS):
    """Return ``{"latest": str, "releases": [str, ...]}`` from PyPI.

    Returns ``None`` when the check is disabled via ``STOCHPYLIB_SKIP_UPDATE_CHECK``
    or the network cannot be reached. Unless ``force=True``, a cache entry
    younger than ``CACHE_TTL_SECONDS`` is served without any network traffic.
    """
    if os.environ.get(SKIP_CHECK_ENV, "").strip() not in ("", "0"):
        return None
    cache = Path(cache_path) if cache_path is not None else _default_cache_path()
    if not force:
        cached = _read_cache(cache)
        if cached is not None:
            return cached
    try:
        import urllib.request

        req = urllib.request.Request(
            PYPI_JSON_URL, headers={"User-Agent": "stochpylib-cli"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    releases = sorted(
        (v for v in data.get("releases", {}) if v),
        key=version_key)
    meta = {
        "latest": str(data.get("info", {}).get("version", "")),
        "releases": releases,
    }
    _write_cache(cache, meta)
    return meta


def install_mode():
    """Classify how stochpylib is installed: ``editable``, ``local`` (direct
    file:// URL), ``wheel``, or ``source`` (no distribution metadata)."""
    try:
        from importlib.metadata import distribution

        dist = distribution("stochpylib")
        raw = dist.read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            if info.get("dir_info", {}).get("editable"):
                return "editable"
            url = str(info.get("url", ""))
            if url.startswith("file://"):
                return "local"
            return "wheel"
        return "wheel"
    except Exception:
        return "source"


def update_available(installed, meta):
    """Compare installed against PyPI latest: ``'update'``, ``'current'``,
    ``'newer'`` (installed is ahead — e.g. an unreleased dev version), or
    ``'unknown'`` (no metadata)."""
    if not meta or not meta.get("latest") or not installed:
        return "unknown"
    delta = version_key(meta["latest"]) > version_key(installed)
    if delta:
        return "update"
    if version_key(meta["latest"]) == version_key(installed):
        return "current"
    return "newer"
