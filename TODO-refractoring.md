# Project TODO — FlexCNN_for_Medical_Physics

Next Plan:
For our next plan, I would like to move some functions out of the jupyter notebook #file:stitching_notebook.ipynb  and into regular python files. I would like to move the following functions: BuildImageSinoTensors(), CNN_reconstruct(), plot_hist_1D(), plot_hist_2D(), and sort_DataSet(), BuildImageSinoTensors() and CNN_reconstruct() can be moved into #file:plot_recons.py , plot_hist1D() and plot_hist_2D() can be moved into #file:plot_histograms.phy , and sort_DataSet() can be moved into #file:sort_dataset.py . When these functions are moved, they may lose access to global variables in the jupyter notebook. Therefore, please look through the functions, update the signatures, and also update the function calls in the notebook. Then, fill in the appropriate imports at the top of each module.


This file lists prioritized, short actionable tasks to improve reproducibility, maintainability, and developer experience.

Priority A — Test the code in the cloud and on a local machine.
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