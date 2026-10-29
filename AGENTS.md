# kldescent

Thesis research package: learns a neural-network drift `U(t,q)` that steers an
initial distribution to a target distribution under underdamped Langevin SDEs,
minimising KL divergence from a free diffusion.

## Layout

- `src/kldescent/` — installable package (hatchling, `src` layout). Entry point is `python -m kldescent` via `src/kldescent/__main__.py`.
  - `model/` — network, normalizing-flow model, training loop, metrics, pretraining
  - `sde/` — Langevin integrators (Euler–Maruyama, BAOAB, Verlet), boundary conditions, losses, terminal costs
  - `benchmark/` — analytic benchmark solutions (Gaussian, overdamped, perturbative)
  - `viz/` — matplotlib plots
  - `utils/` — argparse parsers, logging, seeds, SLURM
- `tests/` — pytest regression tests plus YAML parameter files. The test trains with `tests/gaussian_test_params.yaml` and compares the final position variance against `var_bench`.
- `docs/` — technical markdown (model, training, boundary, benchmark, testing, params).
- Top-level scripts (`evolution_linear.py`, `joint_batch.py`, `fill_points.py`, `test_sample_variance.py`) — standalone analysis/plotting; they import `kldescent` and need the installed package (only `evolution_linear.py` also prepends `src/` to `sys.path`).
- Run outputs (all gitignored) land in top-level folders (`results_42/`, `FIX_s1/`, `res_nonlinear/`, …), each with `data.yaml`, `loss.csv`, `checkpoints/du_checkpoint_*.pt`, `plots_*/` (png/eps/pdf/csv).

## Conventions

- Configuration flows through argparse wrappers in `kldescent.utils.parser` that return plain dicts (`vars(args)`); boundary/integrator/loss/terminal-cost are selected via `getattr(module, args[...])`. No dataclasses for config.
- Tensor state convention: shape `(n, 2d)` — positions in `[:d]`, momenta in `[d:]`.
- NumPy-style docstrings with a Parameters section; short module docstring at the top of each file. No type hints in the package (standalone scripts may use `from __future__ import annotations`).
- YAML for configs and test parameters; run parameters are appended to the run's `data.yaml` log.
- Matplotlib uses the `Agg` backend in scripts; export plots as png + eps + pdf.
- `wandb` logging is optional — guard imports.

## Commands

```bash
uv sync                                   # install (or: pip install -r requirements.txt && pip install -e .)
python -m kldescent --boundary UnderdampedGaussian --bridgetype half --terminalcost MLE --device cpu
pytest tests                              # from repo root (CI sets PYTHONPATH=src, Python 3.12)
```

## Rules

- DO NOT USE pandas.
- No linter or formatter is configured — match the surrounding file's style.
- Local env is Python 3.12 (`.venv`, matches CI); `.python-version` pins 3.13; README says >=3.10.
- Core deps: torch, numpy, scipy, scikit-learn, matplotlib, pyyaml, tqdm, psutil. Optional: wandb (guarded), normflows (only the NormFlow terminal cost).
