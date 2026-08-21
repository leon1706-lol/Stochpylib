# .github/ — CI & publishing

- `workflows/ci.yml` — runs `pytest` on every push/PR across Python 3.10–3.13.
- `workflows/publish.yml` — builds sdist/wheel and publishes to PyPI on `vX.Y.Z` tags via
  Trusted Publisher (OIDC) — no stored API tokens.

To release: tag `vX.Y.Z`, push the tag; CI must be green first.
