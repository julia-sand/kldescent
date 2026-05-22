# kldescent

Code accompanying the paper — learning optimal drift coefficients for underdamped Langevin dynamics via KL divergence minimisation.

The package trains a neural network to approximate the optimal control (drift correction) that steers a Langevin diffusion from an initial distribution to a target distribution, using a stochastic control / diffusion bridge formulation.

## Method overview

The problem is framed as minimising the KL divergence between the controlled path measure and a reference measure. The optimal drift satisfies a Hamilton–Jacobi–Bellman equation whose solution is approximated by a neural network, trained iteratively via stochastic gradient descent on the Girsanov weight along simulated trajectories.


The underlying SDE is underdamped Langevin dynamics:

$$dq = \varepsilon \, p \, dt, \qquad dp = -(p - \sigma_0 \mathbb{U}) \, dt + \sigma_0 \, dW$$

where $\mathbb{U}$ is the learned drift correction and $\sigma_0 = \sqrt{2}$.

## Installation

Requires Python ≥ 3.13. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip, using the pinned requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

PyTorch is installed from the CPU wheel index by default. To use a CUDA build, edit `pyproject.toml` accordingly.

## Running

```bash
python .
```

Command-line arguments select the boundary conditions, bridge type, terminal cost estimator, and device:

```bash
python . --boundary UnderdampedGaussian --bridgetype half --terminalcost MLE --device cpu
```

Key arguments (see `src/kldescent/utils/parser.py` for the full list):

| Argument | Options | Description |
|---|---|---|
| `--boundary` | `UnderdampedGaussian`, `UnderdampedLandauer`, … | Initial/final distribution pair |
| `--bridgetype` | `half`, `marginal` | Bridge formulation |
| `--terminalcost` | `MLE`, `Histogram`, `NormFlow`, `GirsanovNet`, `GirsanovMC` | Terminal cost estimator |
| `--device` | `cpu`, `cuda` | Compute device |
| `--pretrain` | path to CSV | Pre-train network from benchmark drift data |

Results, checkpoints, and metric logs are written to a timestamped folder under the working directory.

## Project structure

```
src/kldescent/
├── sde/
│   ├── boundary.py            # Initial/final distributions and potentials
│   ├── langevin.py            # BAOAB and Euler–Maruyama integrators
│   ├── terminalcost.py        # Terminal cost estimators (half bridge)
│   ├── terminalcost_marginal.py  # Terminal cost estimators (marginal bridge)
│   ├── kl.py                  # KL divergence estimators (kNN)
│   └── girsanovpdf.py         # Girsanov density via Monte Carlo
├── model/
│   ├── nn.py                  # Neural network for drift correction
│   ├── train.py               # Training loop and optimiser
│   ├── metrics.py             # Loss and cumulant metrics
│   ├── pretrain.py            # Pre-training from CSV data
│   └── normflow.py            # Normalising flow density estimator
├── benchmark/
│   ├── gaussian.py            # Non-perturbative BVP solution (Gaussian case)
│   ├── benchmarksol.py        # Benchmark solution wrapper
│   ├── perturbative/          # Perturbative corrections in ε
│   └── overdamped/            # Overdamped reference solutions
├── viz/                       # Plotting utilities for losses, cumulants, distributions
├── utils/                     # Logging, argument parsing, SLURM helpers
├── run.py                     # Entry point
└── trainloop.py               # Single-epoch training logic
```

## Boundary conditions

Boundary condition classes in `sde/boundary.py` define the initial distribution, the reference final distribution, and the associated potential. Available classes:

- `UnderdampedGaussian` — Gaussian initial and final distributions
- `UnderdampedLandauer` — double-well potential final distribution
- `OverdampedFullLandauer`, `OverdampedLandauer` — overdamped variants
- `LinearOU` — linear Ornstein–Uhlenbeck

## Benchmarks

The `benchmark/` module provides reference solutions for comparison:

- **Gaussian BVP** (`gaussian.py`): solves the Hamilton–Jacobi–Bellman equations exactly for Gaussian boundary conditions using `scipy.integrate.solve_bvp`, with epsilon continuation for convergence at large ε.
- **Perturbative solutions** (`perturbative/`): first-order corrections in ε around the free-dynamics solution.

## Dependencies

- [PyTorch](https://pytorch.org/) — neural network and automatic differentiation
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — numerics and BVP solver
- [Matplotlib](https://matplotlib.org/) — visualisation
- [wandb](https://wandb.ai/) — optional experiment tracking
