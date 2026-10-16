# Command-Line Parameters

This file lists all command-line parameters accepted by the package. 

## Training

| Option | Type | Default | Description |
|---|---|---|---|
| `-epochs`, `--epochs` | int | `1` | Total number of training epochs (parameter updates for the drift network). |
| `-logstep`, `--logstep` | int | `1` | Frequency for saving a checkpoint with network parameters and computing loss metrics (measured in epochs). |
| `-lr`, `--lr` | float | `0.0005` | Initial learning rate. |
| `-startdecay`, `--startdecay` | int | `20` | Number of epochs to wait before starting learning-rate scheduling. |
| `-gradclip`, `--gradclip` | float | `0.5` | Global-norm gradient clipping threshold. |
| `-device`, `--device` | str | `cpu` | Compute device. Falls back to `cpu` if a requested GPU is unavailable. |
| `-seed`, `--seed` | int | `1` | Random seed. |

## Problem / Boundary Conditions

| Option | Type | Default | Description |
|---|---|---|---|
| `-boundary`, `--boundary` | str | `UnderdampedLandauer` | Boundary conditions to use (see the `Sde.boundary` module). |
| `-d`, `--d` | int | `1` | Dimension of the position space. |
| `-Tf`, `--Tf` | float | `5` | Final time. |
| `-epsilon`, `--epsilon` | float | `0.2` | Constant $\varepsilon$. |
| `-weight`, `--weight` | float | `20` | Target terminal-cost weight $M_{\text{target}}$. |
| `-bridgetype`, `--bridgetype` | str | `half` | Bridge configuration: `half`, `marginal`, or `full` (applies to the loss function and the benchmark problem). |
| `-muinitial`, `--muinitial` | float | `0` | Initial mean for Gaussian boundary conditions. |
| `-mufinal`, `--mufinal` | float | `0` | Target mean for Gaussian boundary conditions. |
| `-sigmainitial`, `--sigmainitial` | float | `1` | Initial standard deviation for Gaussian boundary conditions. |
| `-sigmafinal`, `--sigmafinal` | float | `0.5` | Target standard deviation for Gaussian boundary conditions. |

## Dynamics and Loss

| Option | Type | Default | Description |
|---|---|---|---|
| `-integrator`, `--integrator` | str | `BAOAB` | SDE integrator: `BAOAB`, `EulerMaruyamaSDE`, or `VerletSDE`. |
| `-batchsamples`, `--batchsamples` | int | `64` | Number of samples used to evaluate the loss at each epoch. |
| `-tsteps`, `--tsteps` | int | `20` | Number of timesteps in the SDE integration. |
|`-testsamples`, `--testsamples` | int | `3` | Number of samples for loss metrics and the terminal cost. |
| `-kramers`, `--kramers` | float | `0` | Kramers coupling constant for regularised dynamics. |
| `-loss`, `--loss` | str | `kl` | Loss function: `kl` (Kullback Leibler divergence). |
| `-terminalcost`, `--terminalcost` | str | `GirsanovMC` | How the terminal cost is evaluated (see the `Sde.terminalcost` module). |

## Model

| Option | Type | Default | Description |
|---|---|---|---|
| `-duhiddenlayerneurons`, `--duhiddenlayerneurons` | int | `16` | Hidden width $H$ of each layer in the drift network (see [model.md](model.md)). |
| `-pretrain`, `--pretrain` | str | `None` | Path (from the current working directory) to a CSV of function values the network is fitted to before training (see `model/pretrain.py` for the CSV structure). Omit if undesired. |

## Metrics, Checkpoints, and Logging

| Option | Type | Default | Description |
|---|---|---|---|
| `-folder`, `--folder` | str | `results2` | Folder where checkpoints and loss metrics are saved. Path is relative to the current working directory. |
| `-checkpointpath`, `--checkpointpath` | str | `None` | Path to an initialisation checkpoint to load at start-up, relative to the current working directory. |
| `-usewandb`, `--usewandb` | str | `None` | WandB project name; omit to disable WandB logging. |
| `-usebenchmark`, `--usebenchmark` | str | `None` | Filepath to a benchmark solution; omit to disable benchmark comparison. |

## Benchmark Cell Problem

| Option | Type | Default | Description |
|---|---|---|---|
| `-cellxnum`, `--cellxnum` | int | `1000` | Number of spatial coordinates in the cell-problem discretisation. |
| `-celltnum`, `--celltnum` | int | `11` | Number of temporal coordinates in the cell-problem discretisation. |
| `-celltfinal`, `--celltfinal` | float | `None` | Final time of the cell problem. If omitted, defaults to the perturbative choice $\epsilon^2$ times the final time. |

## Girsanov Contour Plots

| Option | Type | Default | Description |
|---|---|---|---|
| `-xsamples`, `--xsamples` | int | `20` | Coordinates per axis; total of $x_{\text{samples}}^2$ coordinates. |
| `-mcsamples`, `--mcsamples` | int | `1000` | Monte Carlo trajectories per coordinate. |

## Checkpoint Plots

| Option | Type | Default | Description |
|---|---|---|---|
| `-epoch`, `--epoch` | int | `None` | Epoch of the checkpoint to plot (a file ending in `_epoch` must exist). If omitted, uses the epoch with the lowest action cost. |
| `-mcsamples`, `--mcsamples` | int | `0` | Overrides the number of test samples from the data file when making plots. |