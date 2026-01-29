# Ruff Linting

This project includes a basic `pyproject.toml` Ruff configuration, a `requirements-dev.txt` with `ruff` and `pre-commit`, and a `.pre-commit-config.yaml` to run Ruff linting with `ruff --fix` on commits.

Quick setup:

1. Install development requirements:

```bash
pip install -r requirements-dev.txt
```

2. Install the pre-commit hooks (one-time):

```bash
pre-commit install
```

3. Check all files with Ruff:

```bash
ruff check .
```

4. Auto-fix issues with Ruff:

```bash
ruff check . --fix
```

Adjust `pyproject.toml` to tune rules, ignores, and line-length to fit your project's style.
