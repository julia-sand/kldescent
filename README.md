# kldescent

The package learns an optimal drift that steers an assigned initial distribution towards a target distribution while minimising the Kullback-Leibler divergence from a free diffusion.

The underlying SDE is underdamped Langevin dynamics:

$$dq = \varepsilon p dt, \qquad dp = -(p - \sigma_0 \mathbb{U}) dt + \sigma_0 dW$$

where $\mathbb{U}$ is parametrised by a neural network.

## Installation

Requires Python ≥ 3.10. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip, using the pinned requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

## Running

Requires the package to be installed (see Installation above). Run with:

```bash
python -m kldescent
```

Command-line arguments select the boundary conditions, bridge type, terminal cost estimator, and device:

```bash
python -m kldescent --boundary UnderdampedGaussian --bridgetype half --terminalcost MLE --device cpu
```
## Documentation

Technical details lives in `docs/`:

- [Model Architecture](docs/model.md).
- [Training method](docs/training.md).
- [Boundary Conditions](docs/boundary.md).
- [Benchmark Solution computations](docs/benchmark.md)
- [Regression tests](docs/testing.md)

## Citing

If this repository was useful, please cite 

```@misc{SaLuBaMG2026,
      title={Underdamped Stochastic Optimal Transport with Landauer boundary conditions: machine learning and analytical solution}, 
      author={Julia Sanders and Dario Lucente and Marco Baldovin and Paolo Muratore-Ginanneschi},
      year={2026},
      eprint={2610.41599},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech},
      url={https://arxiv.org/abs/2610.41599}, 
}
```

## AI Disclaimer

For changes made in 2026, the author(s) experimented with several AI tools for accellerating code development, review, documentation, and refactoring. AI generated content was reviewed and edited as needed and the author takes full responsibility for the final content.
