# Repository Guidelines

## Project Structure & Module Organization
The package root `rsl_rl` contains the production code shipped to the grader. Policy network definitions live in `rsl_rl/modules`, PPO and DAGGER variants reside in `rsl_rl/algorithms`, while rollout buffers are in `rsl_rl/storage`. Training entry points are in `rsl_rl/runners`; choose the runner variant that matches the lab deliverable and keep public APIs stable. Environment wrappers and vectorization utilities live in `rsl_rl/env`, and reusable helpers are under `rsl_rl/utils`. Licensing notices are mirrored in `licenses/` and must stay untouched.

## Build, Test, and Development Commands
Create a virtual environment and install the package in editable mode before changing code: `python3 -m venv .venv && source .venv/bin/activate`, then `pip install -e .`. Run a packaging smoke check with `python -c "import rsl_rl; print('loaded')"` to catch dependency issues. When exploring training locally, write thin drivers that instantiate the appropriate `rsl_rl.runners.OnPolicyRunner*` class with your lab configuration.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, keep imports grouped (stdlib, third-party, local), and avoid unused dependencies. Use `snake_case` for functions and variables, `PascalCase` for classes, and reserve ALL_CAPS for constants or configuration keys. Type hints and concise docstrings are expected on new public methods. No formatter is enforced yet; if you introduce `black` or `ruff`, commit the config and run it repo-wide.

## Testing Guidelines
Author unit tests with `pytest` under a mirrored `tests/` hierarchy (for example `tests/modules/test_actor_critic.py`). Cover new loss functions, buffer utilities, and configuration switches. Use `pytest -k pattern` for focused runs, seed stochastic components inside fixtures, and keep tests under one minute so CI and grading scripts remain responsive. Document any long-running experiment checks separately.

## Commit & Pull Request Guidelines
Write commits as short, imperative summaries (e.g., `Add latent critic adapter`). Reference lab tasks with `(#issue)` when relevant and squash fixups locally. Pull requests should explain the scenario, highlight metric changes or TensorBoard screenshots, and list reproduction commands. Call out remaining risks or tuning follow-ups so reviewers can triage quickly.

## Configuration Tips
Keep experiment defaults in lightweight JSON or YAML files tracked with the lab submission. Do not commit large checkpoints; store them in the course-provided artifact bucket instead. Surface new knobs as runner arguments or config keys and document them in the relevant module docstring so downstream agents can reuse them.
