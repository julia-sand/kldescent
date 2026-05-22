# kldescent

Learning optimal drift for underdamped Langevin dynamics via KL divergence minimisation.

The package trains a neural network to approximate the optimal control (drift correction) that steers a Langevin diffusion from an initial distribution towards a target distribution, using a stochastic control / diffusion bridge formulation.

The underlying SDE is underdamped Langevin dynamics:

$$dq = \varepsilon p dt, \qquad dp = -(p - \sigma_0 \mathbb{U}) dt + \sigma_0 dW$$

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

Results, checkpoints, and metric logs are written to a timestamped folder under the working directory.

## Boundary conditions

Boundary condition classes in `sde/boundary.py` define the initial distribution, the reference final distribution, and the associated potential. Available classes:

- `UnderdampedGaussian` — Gaussian initial and target distributions
- `UnderdampedLandauer` — double-well potential target distribution

## Benchmarks

The `benchmark/` module provides reference solutions for comparison:

- **Gaussian BVP** (`gaussian.py`): solves the Hamilton–Jacobi–Bellman equations exactly for Gaussian boundary conditions using `scipy.integrate.solve_bvp`, with epsilon continuation for convergence at large ε.
- **Perturbative solutions** (`perturbative/`): first-order corrections in ε around the free-dynamics solution.
