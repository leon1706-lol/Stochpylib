# Security Policy

## Supported versions

stochpylib is pre-1.0; only the latest released version receives security fixes.

| Version | Supported |
|---------|-----------|
| latest on PyPI | yes |
| older releases | no |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Use one of these private channels instead:

1. **Preferred:** GitHub's *Report a vulnerability* button (Security tab of this repository).
2. Email the maintainer at **leonschwarzkopf08@gmail.com** with `[security]` in the subject.

Include as much of the following as you can: affected version(s) (`spl --version`), a minimal
reproduction, impact assessment, and any suggested fix. Please give us a reasonable window to
release a fix before any public disclosure.

## What to expect

- Acknowledgment within **72 hours**.
- An assessment and (if accepted) a fix plan within **7 days**.
- A patched release; you will be credited in the release notes unless you prefer otherwise.

## Scope notes

This library processes user-supplied numerical data locally and performs no network I/O, holds
no secrets, and installs no scripts beyond the `spl` console entry point — so most classic
CVE classes do not apply. Vulnerabilities of interest include: code execution via crafted
inputs (e.g. pathological array shapes/types causing unsafe behavior), dependency-level
issues in NumPy/SciPy usage patterns we control, and supply-chain issues in our build/publish
workflows.

Dependency scanning happens implicitly via CI; reports about transitive NumPy/SciPy
vulnerabilities are best filed upstream, but feel free to notify us if a pinned workaround is
needed.
