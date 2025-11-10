# Project TODO — FlexCNN_for_Medical_Physics

This file lists prioritized, short actionable tasks to improve reproducibility, maintainability, and developer experience.

Priority A — high (do these first)
- Add README.md
  - Purpose: short project description, quickstart, example commands.
  - Include: install (editable install), dataset format (shapes/dtypes/layout), minimal example run using a tiny/dummy dataset.

- Add dependency manifest
  - Create `requirements.txt` with pinned versions or `pyproject.toml` (PEP 621).
  - Include dev deps: `pytest`, `ruff`/`flake8`, `black`, `isort`.

- Add minimal tests + CI
  - Add `tests/` with:
    - test_dataset_loading.py: instantiate `classes/dataset.py` on tiny dummy files/arrays.
    - test_model_forward.py: create minimal generator/discriminator and run a single forward pass with a small random tensor.
  - Add GitHub Actions workflow `.github/workflows/ci.yml` to run tests and lint on push/PR.

Priority B — medium
- Add formatting/lint config
  - Add `pyproject.toml` or `setup.cfg` with `black`/`ruff`/`isort` settings to enforce consistent style.

- Consolidate style guidance
  - Keep the project-level `.github/copilot-instructions.md` as canonical.
  - Remove or align any conflicting guidance in other folders.

- Add `CONTRIBUTING.md` and `LICENSE`
  - Provide developer onboarding steps, testing, formatting, and PR guidelines.
  - Choose and add an appropriate open-source license file.

Priority C — nice-to-have
- Add `examples/` or `notebooks/` with a tiny end-to-end notebook or script that trains on synthetic data for 1-2 iterations.
- Add `docs/` or a short `docs/README.md` describing config dicts in `config_net_dicts/`.
- Add performance guidance: memory/patching recommendations, reproducible RNG seeding, deterministic augmentation examples.

Technical checks to add (short)
- Verify that data loaders in `classes/dataset.py`:
  - Support lazy loading or tiling (not loading entire dataset into RAM) for large medical images.
  - Expose deterministic seeding for augmentations.
- Add a small memory OOM guard or recommended default patch sizes in the config files.

Suggested quick commands (developer)
- Install editable package:
  - `pip install -e .`
- Run tests:
  - `pytest -q`
- Format and lint:
  - `black .`
  - `ruff . --fix`