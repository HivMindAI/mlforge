# AGENTS.md

Durable working rules for MLForge contributors and coding agents.

## Collaboration

- Work one milestone at a time.
- Do not start the next milestone until the project owner explicitly says `DONE`.
- Explain important concepts clearly before implementing them.
- Keep changes small, reviewable, and limited to the active milestone.
- Preserve useful existing work and avoid unrelated refactoring.
- Do not commit, push, merge, force-push, reset, or rewrite history unless explicitly asked.

## Repository Safety

- Inspect `git status`, the active branch, and relevant files before editing.
- Never commit secrets, credentials, local `.env` files, datasets, model artifacts, caches, or virtual environments.
- Document required environment variables in `.env.example`.
- Keep production logic in importable Python modules, not notebooks.
- Prefer `src/mlforge` until there is a concrete reason to reorganize.

## Python Standards

- Use Python 3.11 or newer.
- Prefer type hints, descriptive names, pathlib, clear errors, and testable functions.
- Avoid import-time side effects, hidden global state, mutable defaults, bare `except`, and data leakage in ML code.
- Add dependencies only when the current milestone requires them.
- Keep production dependencies minimal.

## Testing And Validation

After Python changes, run the checks configured in the repository. At minimum, run:

```powershell
ruff check .
ruff format --check .
mypy src tests
python -m pytest
python -m build
```

If formatting fails, run:

```powershell
ruff format .
ruff check .
ruff format --check .
mypy src tests
python -m pytest
python -m build
```

Do not claim checks passed unless they were actually run.

## Teaching Focus

For important concepts, explain:

- What it is.
- Why MLForge needs it.
- How it works.
- Where it belongs in the architecture.
- Common mistakes.
- How it is tested.
- How the development version differs from a production version.

## Frontend Validation

The frontend starts only in its milestone. After every frontend change, run:

```powershell
npm run lint
npm run build
```

Also run frontend tests when a test command exists.
