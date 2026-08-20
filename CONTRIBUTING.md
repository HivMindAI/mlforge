# Contributing to MLForge

MLForge is an alpha-stage local tabular-ML toolkit developed in deliberately small milestones.
Before changing code, read `README.md`, `ROADMAP.md`, `docs/architecture.md`, and `AGENTS.md`. Keep
each contribution within the accepted product scope and avoid infrastructure or abstractions for
unaccepted future work.

Security vulnerabilities follow [SECURITY.md](SECURITY.md), not the normal public issue workflow.

## Development setup

MLForge requires Python 3.11 or newer. From a repository clone:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The editable install is important: MLForge uses a `src/` layout, and tests validate installed
distribution metadata and entrypoints. The development extra caps NumPy below 2.5 so a Python 3.12
development environment can type-check MLForge against its supported Python 3.11 language target;
this does not narrow the runtime dependency selected through pandas and scikit-learn.

## Required checks

Run the same checks as CI before requesting review:

```powershell
ruff check .
ruff format --check .
mypy src tests
python -m pytest
python -m build
python -m twine check --strict dist/*
```

If formatting fails, run `ruff format .`, then repeat the full check sequence. Do not weaken a check
or test merely to make the suite green. `python -m pytest` enforces the project's 80% statement
coverage floor.

## Change guidelines

- Put production behavior in importable modules under `src/mlforge`, not notebooks or CLI handlers.
- Keep the CLI thin: parse and present at the boundary, while domain modules own behavior.
- Use explicit typed inputs, return values, and domain exceptions.
- Avoid import-time logging configuration, environment reads, filesystem writes, or global
  registration.
- Add dependencies only when the active milestone exercises them.
- Add behavioral tests for success, failure, and important edge cases.
- Update documentation whenever a public interface or supported capability changes.
- Treat names listed in a domain module's `__all__` as public. Update `docs/api.md`, compatibility
  notes, and interface tests for intentional additions, removals, or signature changes.
- Decide whether a serialized manifest change requires a new schema version; never make a reader
  guess between incompatible shapes.
- Do not commit credentials, `.env` files, datasets, fitted models, caches, or experiment output.

## Pull requests

A reviewable pull request should explain the user or architecture problem, the chosen boundary, the
tests added, and the commands actually run. Keep unrelated refactoring out of the change. Do not
claim support for behavior that is still planned.

CI runs on Ubuntu with Python 3.11/3.12 and Windows with Python 3.12. It additionally builds the
wheel, installs it into a separate virtual environment, runs `pip check`, and executes
`scripts/wheel_smoke.py` outside the repository. This confirms that the distribution—not an
editable source path—supports the documented local workflow.

## Contribution license

MLForge is licensed under the [Apache License 2.0](LICENSE). Unless explicitly stated otherwise,
an intentionally submitted contribution is provided under those terms, as described by section 5
of the license. Do not submit code or data you do not have the right to contribute.
