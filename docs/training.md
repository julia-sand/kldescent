# Training Summary

This file summarises the gradient descent process by which the neural network is trained. 

## 1. Training Objective (Loss Function)

The training objective is defined in `src/kldescent/sde/loss.py`. 

### Stochastic Differential Equation

The underdamped dynamics are 

$$dq = \varepsilon p dt, \qquad dp = -(p - \sigma_0 \mathbb{U}) dt + \sigma_0 dW$$

with parameters $\varepsilon$ given by `--epsilon` and $\sigma_0=\sqrt{2}$

### Loss Function

For a batch of initial points, the stochastic differential equation is evolved and the loss $\hat{L}$ is accumulated along trajectories, according to:

$$\hat{L} = \mathbb{E}\left[ 0.5 a + (0.5 \bar{a} + g) \cdot c \right]$$

with the terms:

- $a$, the integrated squared norm of the learned drift:

$$a = \int \|\mathbb{U}^{\Theta}\|^2 dt$$

- $\bar{a}$, the same integral with gradients detached (accumulated under `torch.no_grad()`):

$$\bar{a} = \int \|\mathbb{U}^{\bar{\Theta}}\|^2 dt$$

- $c$, the stochastic integral of the drift against the Wiener process:

$$c = \int \mathbb{U}^{\Theta}\cdot dw$$

- $g$, the terminal cost penalizing the distance between the current final distribution and the target, see [Terminal Cost](training.md#4-terminal-cost).

Evolutions of the stochastic differential equation are calculated with a stratified random time grid (`stratified_time_grid`). This splits the interval into three intervals: $[0, 0.2 T_f]$, $[0.8 T_f, T_f]$, and $[0.2 T_f, 0.8 T_f]$ with approximately one third of the total number of timesteps `--tsteps` in each interval. This means that the timegrid is more concentrated around the boundary layers, where the dynamics are most non-linear.

## 2. Numerical Integration Schemes

The trajectories are integrated numerically with discretisation schemes specified in `src/kldescent/sde/langevin.py`.The scheme used for integration can be specified by `--integrator`. For a $2d$ dimensional phase space, the positions are indexed by `[:d]` and momenta are indexed by `[d:]`. 

We denote the time grid by $t_0<t_1<\ldots<t_{N-1}<t_N = T$ in the following. Initial positions and momenta are sampled from the assigned initial distribution. 

### Euler-Maruyama Scheme (`EulerMaruyamaSDE`)

\begin{align*}
q_{t_{n+1}} &= q_{t_{n}} + \varepsilon p_{t_{n}} (t_{n+1}-t_{n}) \\
p_{t_{n+1}} &= p_{t_{n}} - (p_{t_{n}} - \sigma_0 \mathbb{U}(t_n, q_{t_{n}}))(t_{n+1}-t_{n}) + \sigma_0 \Delta W_n,
\end{align*}

where $\Delta W_n \sim \mathcal{N}(0, t_{n+1}-t_{n})$.

### BAOAB Scheme (`BAOAB`)

An operator-splitting (Strang) method for the Langevin dynamics. This tends to be more stable and accurate than the Euler-Maruyama scheme. [1]
- **B (Momentum)**: deterministic momentum kick (half-step):

$$p \gets p + \sigma_0\mathbb{U}(t_{n}, q)\,(t_{n+1} - t_{n})/2 $$

- **A (Position)**: position half-step:

$$q \gets q + \varepsilon p\, (t_{n+1} - t_{n})/2$$

- **O (Ornstein-Uhlenbeck)**: exact solution of the linear OU dynamics for the momenta:

$$p \gets e^{-(t_{n+1} - t_{n})} p + \sigma_0 \tilde{W}, \qquad \tilde{W} \sim \mathcal{N}\left(0, \frac{1 - e^{-2(t_{n+1} - t_{n})}}{2}\right).$$

- **A (Momentum)**: position half-step.

$$q \gets q + \varepsilon p\, (t_{n+1} - t_{n})/2$$

- **B (Position)**: final drift kick (half-step) with the drift evaluated at the new state.

$$p \gets p + \sigma_0 \mathbb{U}(t_{n+1}, q)\, (t_{n+1} - t_{n})/2 $$

### Verlet Scheme (`VerletSDE`)

A symmetric integrator with position half-steps around a full momentum update
- **A (Position)**: position half-step:

$$q \gets q + \varepsilon p\, (t_{n+1} - t_{n})/2$$

- **B (Momentum)**: momentum update:

$$p \gets p + \sigma_0\mathbb{U}(t_{n+1}, p)\,(t_{n+1} - t_{n}) + \sigma_0 \Delta W_n $$

- **A (Momentum)**: position half-step.

$$q \gets q + \varepsilon p\, (t_{n+1} - t_{n})/2$$

## 3. Terminal Cost

The terminal cost $g$ is estimated by one of the classes in `src/kldescent/sde/terminalcost.py`. The method used to compute the terminal cost can be specified using the `--terminalcost` flag.  

- **GirsanovMC** (default): estimates the final joint density $\hat{\rho}_T(q,p)$ by a Monte Carlo integration of the reversed dynamics (`compute_log_pdf` in `sde/girsanovpdf.py`) with `--testsamples` trajectories from each terminal point.
- **MLE**: For Gaussian boundary conditions only. Assume that the final distribution is Gaussian and use the sample mean and covariance of samples from the final distribution into a closed form.
- **Histogram**: 2D histogram fit with samples from the final distribution.
- **NormFlow**: a normalising flow (`RhoNormFlow`) fit with samples from the final distribution. 
- **GirsanovNet**: a neural network (`RhoNeuralNetwork` + `TrainRho`) trained on the current final distribution.

The terminal cost is

$$g(x) = M \Big( \log \hat{\rho}_T(q_T,p_T) + \log \rho_{\text{target}}(q_T,p_T) \Big),$$

The constant $M$ is set by `--weight` (default 20). 

## 4. Training Process

### Training Loop

Training is controlled by `train_loop_du` in `src/kldescent/trainloop.py`, with the gradient update steps calibrated in the `Train` class in `src/kldescent/model/train.py`. One epoch of training corresponds to:

1. **Terminal cost update**: the terminal cost estimator refreshes its estimate of the final distribution $M$ ([Terminal Cost](training.md#3-terminal-cost)) (note: may be done later, depending on the method).
2. **Trajectory Rollout and Loss**: The loss $\hat{L}$ is computed ([Training Objective](training.md#1-loss-function)).
3. **Gradient Computation**: gradients are backpropagated through the SDE trajectories to the network weights. If the loss is NaN/Inf, this step is skipped and the training skips this batch.
4. **Parameter Update**:
    - An `AdamW` optimizer with weight decay $10^{-3}$ updates the parameters in the feed-forward layers. [2] 
    - **Learning Rate**: initial value set by `--lr` (default: `0.0005`).
    - **Gradient Clipping**: global norm clipping with threshold `--gradclip` (default: `0.5`). [3]
5. **Learning Rate Scheduling**: `CosineAnnealingLR` decays the learning rate (floor $10^{-10}$, $T_{\max} =$ epochs $-$ `startdecay`). The scheduling starts after `--startdecay` epochs have already passed (default: `20`). [4]

### Pre-training

If a CSV is provided via `--pretrain` as a path to the CSV, `run_pretrain` in `src/kldescent/model/pretrain.py` fits the network to the function values, as an initialisation for the network training.  

## 5. Metrics and Logging

Metrics are computed to track training progress:
- the dynamic KL cost
$$\text{kldynamic} = \frac{1}{2}\,\mathbb{E}\int\|\mathbb{U}\|^2 dt$$
is accumulated along test trajectories for a batch of test samples;
- static KL divergences between the current final distribution and the target, and between the current final distribution and the benchmark (when `--usebenchmark` is given, otherwise the final distribution of a free diffusion), are estimated by a k-nearest-neighbour estimator (`src/kldescent/sde/kl.py`) for non-Gaussian boundary conditions; for the UnderdampedGaussian case, we use a close-form expression based on sample means and covariances.
- the action cost
$$\text{action} = \text{kldynamic} + M \text{klstatic}$$

## 6. References 

[1] Leimkuhler & Matthews (2013), "Rational construction of stochastic numerical methods for molecular sampling"
[2] Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization", ICLR, arXiv:1711.05101 — AdamW.
[3] Pascanu, Mikolov & Bengio (2013), ICML, PMLR 28(3), 1310–1318
[4] Loshchilov & Hutter (2017), "SGDR", ICLR, arXiv:1608.03983
