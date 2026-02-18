# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python Discord bot with a single runtime module:

- `main.py`: bot startup, config loading, queue worker, and `pi_sdk` integration.
- `bot.config.toml.example`: template for local runtime config.
- `bot.config.toml`: local secrets/runtime settings (ignored by git).
- `.pi_sessions/`: runtime session state (ignored by git).
- `README.md`: setup and behavior notes.

Keep new code close to the current layout unless a clear module split is needed.

## Build, Test, and Development Commands
Use `uv` for dependency management and command execution.

- `uv sync`: create/update the project environment from `pyproject.toml` and `uv.lock`.
- `cp bot.config.toml.example bot.config.toml`: create local config.
- `uv pip install -e ~/Developer/pi_sdk` (or `~/Developer/pi-py`): install local SDK dependency.
- `uv run python3 main.py`: run the bot.
- `uv run python3 -m py_compile main.py`: quick syntax check before opening a PR.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and type hints for public/internal functions.
- Use `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, `PascalCase` for classes/dataclasses.
- Prefer small helper functions for config validation, matching existing patterns like `_get_required_string`.
- Keep logging structured and actionable (`LOGGER.info(...)` with IDs/context).

## Testing Guidelines
There is no automated test suite configured yet. For now:

- Run `uv run python3 -m py_compile main.py` and perform a local bot smoke test.
- For new logic-heavy helpers, add focused unit tests in a new `tests/` directory using `test_*.py` naming.
- Prioritize coverage for config parsing, queue behavior, and session TTL logic.

## Commit & Pull Request Guidelines
Git history uses Conventional Commit prefixes (`feat:`, `fix:`, `chore:`). Continue that format.

- Keep commits small and scoped to one behavior change.
- PRs should include: summary, why the change is needed, config/env updates, and manual verification steps.
- Link related issues and include logs or screenshots when behavior/output changes.
