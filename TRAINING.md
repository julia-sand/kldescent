# Network Training Overview

The network is trained using a method based on KL-divergence descent to learn a non-perturbative drift (parameterized by a neural network) that bridges an initial distribution to a target terminal state. The training entry point is `kldescent.run.run()`, which parses the command line, builds the boundary conditions, terminal cost, network, integrator, loss and trainer, and drives the epoch loop.

## 1. Model Architecture
The drift is parameterized by the `NeuralNetwork` class (`src/kldescent/model/nn.py`), which uses a multi-branch architecture combining feed-forward networks, Fourier feature encoding, and a terminal score anchor.

### Normalisation
Inputs are rescaled with boundary-dependent constants held in `MinMaxNorm_layers` (non-trainable buffers):
- Time: $\hat{t} = t / T_f$
- Position: $\hat{x} = (x + \|\text{xmax}\|) / (2 \|\text{xmax}\| + 10^{-8})$, where `xmax` is `boundary.norm_qmax` (e.g. 3.5 for `UnderdampedLandauer`, 5 for `UnderdampedGaussian`).

### Network Branches
1.  **Time Branch (`NNt`)**:
    - A feed-forward network: $\hat{t} \to \text{Linear} \to \text{GELU} \to \text{Linear} \to \text{GELU} \to \mathbb{R}^H$.
2.  **State Branch (`NNx`)**:
    - A feed-forward network: $\hat{x} \to \text{Linear} \to \text{GELU} \to \text{Linear} \to \mathbb{R}^H$.
3.  **Fourier Branch (`FourierFeatures`)**:
    - A high-frequency encoding of time: $F(\hat{t}) = [\sin(2\pi B\hat{t}), \cos(2\pi B\hat{t})] \in \mathbb{R}^{2H}$, where $B \in \mathbb{R}^{H \times 1}$ is a **trainable** linear layer without bias, initialised $\mathcal{N}(0, \text{scale}^2)$. The scale is problem-dependent:
    $$\text{scale} = 0.3 \, T_f \sqrt{1 + 4\epsilon^2}.$$

### Integration and Output
- **Combination**: The outputs of the time and state branches are concatenated and passed through a `LayerNorm` layer: $c = \text{LayerNorm}([T(\hat{t}), X(\hat{x})]) \in \mathbb{R}^{2H}$.
- **Residual Blocks (`NN1`, `NN2`)**: The network uses two residual-like blocks (`NN2` modules). Each block performs:
  $$\text{NN2}(v) = L_3(\text{GELU}(v_2 + \text{GELU}(L_2(v_2)))), \quad v_2 = \text{GELU}(L_1(v))$$
  mapping $\mathbb{R}^{2H} \to \mathbb{R}^H \to \mathbb{R}^H \to \mathbb{R}^1$.
- **Final Output**: The final drift $\mathbb{U}(t,x)$ is the sum of a learned component and an anchoring term based on the target distribution's score:
  $$\boxed{\mathbb{U}(t,x) = \text{NN1}(c) + \text{NN2}\!\bigl(F(\hat{t})\bigr) \cdot \text{scorefinal}(x)}$$
  where `scorefinal` is the negative log-gradient of the target distribution provided by the boundary class. This anchors the network to the score of the target distribution at all times.

## 2. Training Objective (Loss Function)
The training objective is defined in `src/kldescent/sde/loss.py`. The primary loss function used is the **KL-divergence loss** (the `kl` class, based on Proposition 3 of arXiv:2410.05163). The loss is selected with `--loss` (default `kl`; `ep` is a placeholder).

### Loss Formulation
The loss $\hat{L}$ is computed by rolling out trajectories of the SDE:
$$\hat{L} = \mathbb{E}\left[ 0.5 a + (0.5 \bar{a} + g) \cdot c \right]$$
Where:
- $a = \int \|du\|^2 dt$: The integrated squared norm of the learned drift.
- $\bar{a} = \int \|du_{\text{detached}}\|^2 dt$: The same integral but with gradients detached (accumulated under `torch.no_grad()`).
- $c = \int du \cdot dw$: The stochastic integral of the drift against the Wiener process.
- $g$: The terminal cost from the terminal cost class (specified by `--terminalcost`, see Section 4), which penalizes the distance between the current final distribution and the target.

The rollout uses a **stratified random time grid** (`stratified_time_grid`): roughly one third of the `--tsteps` points in $[0, 0.2 T_f]$, one third in $[0.8 T_f, T_f]$, and one third in $[0.2 T_f, 0.8 T_f]$, plus the endpoints. This concentrates resolution on the boundary layers where the dynamics are most non-linear.

## 3. Numerical Integration Schemes
The trajectories are integrated numerically with the schemes in `src/kldescent/sde/langevin.py` (selected by `--integrator`). All schemes integrate the underdamped Langevin system
$$dq = \varepsilon p\, dt, \qquad dp = -(p - \sigma_0\,\mathbb{U})\,dt + \sigma_0\,dW,$$
with state stored as positions `[:d]` and momenta `[d:]`.

### Euler-Maruyama Scheme (`EulerMaruyamaSDE`)
The simplest numerical method: given the drift evaluated at $(t_n, X_n)$,
$$q_{n+1} = q_n + \varepsilon p_n \Delta t, \qquad p_{n+1} = p_n - (p_n - \sigma_0 du_n)\Delta t + \sigma_0 \Delta W_n,$$
where $\Delta W_n \sim \mathcal{N}(0, \Delta t)$.

### BAOAB Scheme (`BAOAB`)
An operator-splitting (Strang) method for the Langevin dynamics, more stable and accurate than Euler-Maruyama for larger timesteps. One step is the "sandwich" B-A-O-A-B:
- **B (Drift)**: deterministic momentum kick (half-step): $p \gets p + \frac{\sigma_0}{2} du\, \Delta t$.
- **A (Position)**: position half-step: $q \gets q + \varepsilon p\, \Delta t/2$.
- **O (Ornstein-Uhlenbeck)**: exact solution of the linear OU dynamics for the momenta: $p \gets e^{-\Delta t} p + \sigma_0 \tilde{W}$, with innovation $\tilde{W} \sim \mathcal{N}(0, (1 - e^{-2\Delta t})/2)$.
- **A**: another position half-step.
- **B**: final drift kick (half-step) with the drift evaluated at the new state.

In the loss rollout (`loss_integration_step`) the second B kick is deferred: the last half-step of each step uses the drift computed at the *next* time, and the very last B half-step of the whole trajectory is completed in `loss_final_step` so the terminal cost is evaluated with the fully integrated state.

### Verlet Scheme (`VerletSDE`)
A symmetric integrator with position half-steps around a full momentum update
$$p_{n+1} = p_n - (p_n - \sigma_0 du_n)\Delta t + \sigma_0 \Delta W_n.$$

## 4. Terminal Cost
The terminal cost $g$ is estimated on the fly by one of the classes in `src/kldescent/sde/terminalcost.py` (for `--bridgetype half`, the default) or `src/kldescent/sde/terminalcost_marginal.py` (for `--bridgetype marginal`). Each estimator is updated every epoch by `update_cost_params(integrator, epoch)` at the start of the training loop.

- **GirsanovMC** (default): estimates the final phase-space density $\hat{P}(q,p)$ via a backward Girsanov Monte Carlo integration (`compute_log_pdf` in `sde/girsanovpdf.py`) with `--testsamples` trajectories per terminal point.
- **GirsanovNet**: a neural network (`RhoNeuralNetwork` + `TrainRho`) trained on the current final distribution.
- **MLE**: parametric bivariate-Gaussian estimate from sample cumulants of the final state.
- **Histogram**: 2D histogram of the final positions/momenta.
- **NormFlow**: a normalising flow (`RhoNormFlow`) fit to the final distribution.

The terminal cost is
$$g(x) = M \Big( \log \hat{P}(x) + \log p_T(x) \Big) = M \Big( \log \hat{P}(x) - \log p_{\text{ref}}(x) \Big),$$
where $M$ is ramped in over the first 30 epochs: $M = M_{\text{target}} \min(1, (epoch/30)^2)$ with $M_{\text{target}}$ set by `--weight` (default 20). This gradually turns on the terminal penalty while the running cost drives the initial adaptation. For the marginal bridge type the same estimators act on the position marginal with $g(q) = M(\log \hat{p}(q) + U_{\text{final}}(q))$.

## 5. Boundary Conditions
Boundary conditions define the initial and final distributions. They are implemented in `src/kldescent/sde/boundary.py` and selected with `--boundary`.

### UnderdampedLandauer
Initial and target distributions modelling Landauer's bit erasure:
- **Initial Distribution**: $p_0(q, p) = p_0(q) \mathcal{N}(p; 0, 1)$
- **Target Distribution**: $p_T(q, p) = p_T(q) \mathcal{N}(p; 0, 1)$

with position marginals
$$p_0(q) \propto e^{-(q - 1)^4}, \quad p_T(q) \propto e^{-(q^2 - 1)^2}$$
and volatility $\sigma_0 = \sqrt{2}$.

### UnderdampedGaussian
Both the initial and target distributions are Gaussians:
- **Initial Distribution**: $\mathcal{N}(\mu_{\text{initial}}, \sigma_{\text{initial}}^2)$
- **Target Distribution**: $\mathcal{N}(\mu_{\text{final}}, \sigma_{\text{final}}^2)$

The parameters are passed via `gaussian_params()` (i.e. `--muinitial`, `--mufinal`, `--sigmainitial`, `--sigmafinal`). This is useful for testing the bridge between Gaussian states in an underdamped system, with a benchmark solution available in `benchmark/`.

Other boundary options include `OverdampedLandauer`, `OverdampedFullLandauer` (delta initial state), and `LinearOU` (linear Ornstein-Uhlenbeck with analytic solution).

## 6. Training Process

### Training Loop
Training is managed by `Train` in `src/kldescent/model/train.py` and orchestrated by `train_loop_du` in `src/kldescent/trainloop.py`.

1. **Terminal cost update**: the terminal cost estimator refreshes its estimate of the final distribution and ramps $M$ (Section 4).
2. **Trajectory Rollout and Loss**: trajectories are integrated from the initial boundary distribution using the selected integrator, and the loss $\hat{L}$ is computed (Section 2).
3. **Gradient Computation**: gradients are backpropagated through the SDE trajectories to the network weights. If the loss is NaN/Inf the step is skipped.
4. **Parameter Update**:
    - A single `AdamW` optimizer (weight decay $10^{-3}$) updates the feed-forward layers. The Fourier feature weights are currently **frozen** during training (the second `AdamW` optimizer for `FourierFeatures` is disabled in `train.py`).
    - **Learning Rate**: initial value set by `--lr` (default: `0.0005`).
    - **Gradient Clipping**: global norm clipping with threshold `--gradclip` (default: `0.5`).
5. **Learning Rate Scheduling**: `CosineAnnealingLR` decays the learning rate (floor $10^{-10}$, $T_{\max} =$ epochs $-$ `startdecay`), stepped only after `--startdecay` epochs (default: `20`).

### Metrics and Logging
Every `--logstep` epochs (`Metrics.update_metrics` in `src/kldescent/model/metrics.py`):
- the dynamic KL cost $\frac{1}{2}\mathbb{E}\int\|\mathbb{U}\|^2 dt$ is accumulated along test trajectories;
- static KL divergences between the current final distribution, the reference, and the benchmark (when `--usebenchmark` is given) are estimated with a k-nearest-neighbour estimator (`src/kldescent/sde/kl.py`);
- the action cost $\text{action} = \text{kl\_dynamic} + \texttt{weight}\cdot\text{kl\_static}$, the loss and the current learning rate are written to the results folder, and optionally logged to Weights & Biases (`--usewandb`).
A checkpoint of the drift network is saved each logged epoch (`checkpoints/du_checkpoint_<epoch>.pt`), and an initial checkpoint is saved before training starts. A checkpoint for resuming can be loaded with `--checkpointpath`.

### General Configuration
Key arguments (see `src/kldescent/utils/parser.py` for the full list):
- `--epochs` (default: `1`): total number of training epochs.
- `--logstep` (default: `1`): checkpoint/metric logging frequency in epochs.
- `--folder` (default: `results2`): results folder (a timestamped subfolder is created).
- `--device` (default: `cpu`): compute device.
- `--batchsamples` (default: `64`): samples in each training loss rollout.
- `--tsteps` (default: `20`): number of timesteps in the SDE integration.
- `--testsamples` (default: `3`): samples for metrics/terminal cost.
- `--boundary`, `--bridgetype` (default: `half`), `--terminalcost` (default: `GirsanovMC`), `--integrator` (default: `BAOAB`), `--loss` (default: `kl`).
- `--d` (default: `1`), `--Tf` (default: `5`), `--epsilon` (default: `0.2`), `--weight` (default: `20`).
- `--duhiddenlayerneurons` / `--rhohiddenlayerneurons` (default: `16`): hidden width $H$.

### Pre-training
If pre-training data is provided (via `--pretrain`, a CSV path), `run_pretrain` in `src/kldescent/model/pretrain.py` fits the network to the tabulated function values (scaled by `epsilon` when supplied) before the KL-descent process begins.

## 7. Key Components Summary

| Component | Location | Responsibility |
| :--- | :--- | :--- |
| `NeuralNetwork` | `src/kldescent/model/nn.py` | Parametrizes the drift $\mathbb{U}(t, x)$. |
| `kl` (Loss) | `src/kldescent/sde/loss.py` | Computes the KL-divergence based loss via SDE rollout. |
| Integrators (`EulerMaruyamaSDE`, `BAOAB`, `VerletSDE`) | `src/kldescent/sde/langevin.py` | Numerical integration of the Langevin SDE for loss rollouts and metrics. |
| `TerminalCost` + estimators | `src/kldescent/sde/terminalcost.py`, `src/kldescent/sde/terminalcost_marginal.py` | Estimates the terminal penalty $g$ from the current final distribution. |
| Boundaries | `src/kldescent/sde/boundary.py` | Initial/target distributions, sampling, and `scorefinal` anchor. |
| `Train` | `src/kldescent/model/train.py` | Optimizer, gradient clipping, LR schedule, `train_step`. |
| `train_loop_du` | `src/kldescent/trainloop.py` | Per-epoch orchestration: terminal-cost update, training step, checkpointing. |
| `Metrics` | `src/kldescent/model/metrics.py` | KL metrics, action cost, logging (CSV / WandB). |
| `Pretrain` / `run_pretrain` | `src/kldescent/model/pretrain.py` | Optional CSV-based pre-training of the drift network. |
