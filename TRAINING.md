# Network Training Overview

The network is trained using a method based on KL-divergence descent to learn a non-perturbative drift (parameterized by a neural network) that bridges an initial distribution to a target terminal state.

## 1. Model Architecture
The drift is parameterized by the `NeuralNetwork` class, which combines Fourier feature encoding for time and a feed-forward MLP for the state.

### Fourier Feature Encoding
For the time variable $t$, a `FourierFeatures` module is used to project the scalar time input into a higher-dimensional space. 
- **Projection**: $t \to 2\pi B t$, where $B$ is a learnable linear layer (`nn.Linear`) initialized with a normal distribution (standard deviation defined by `scale`).
- **Encoding**: The projected values are passed through $\sin(\cdot)$ and $\cos(\cdot)$ functions, concatenating the results to form the Fourier embedding.

### Feed-Forward Network (`FeedForwardLayers`)
The encoded time and the state $x$ are concatenated and passed through a deep MLP:
- **Input**: Concatenation of the Fourier-encoded time and the current state $x$.
- **Layers**: A `nn.Sequential` stack consisting of:
    - Multiple linear layers (`nn.Linear`) with a hidden size defined by `--duhiddenlayerneurons` (default: `16`).
    - **Activation Function**: `nn.Tanh()` is used between all linear layers.
- **Output**: A final linear layer that outputs the drift $\text{du}(t, x)$.

### Overall Flow
$$\text{Input}(t, x) \to \text{Fourier}(t) \oplus x \to \text{MLP} \to \text{du}(t, x)$$

## 2. Training Objective (Loss Function)
The training objective is defined in `src/kldescent/sde/loss.py`. The primary loss function used is the **KL-divergence loss** (based on Proposition 3 of arXiv:2410.05163).

### Loss Formulation
The loss $\hat{L}$ is computed by rolling out trajectories of the SDE:
$$\hat{L} = \mathbb{E}\left[ 0.5 a + (0.5 \bar{a} + g) \cdot c \right]$$
Where:
- $a = \int \|du\|^2 dt$: The integrated squared norm of the learned drift.
- $\bar{a} = \int \|du_{\text{detached}}\|^2 dt$: The same integral but with gradients detached to stabilize learning.
- $c = \int du \cdot dw$: The stochastic integral of the drift against the Wiener process (Brownian motion).
- $g$: The terminal cost calculated by the `TerminalCost` class (specified by `--terminalcost`), which penalizes the distance between the final distribution and the target.

## 3. Numerical Integration Schemes
The trajectories are integrated numerically using schemes defined in `src/kldescent/sde/langevin.py` (specified by `--integrator`).

### Euler-Maruyama Scheme (`EulerMaruyamaSDE`)
The Euler-Maruyama scheme is the simplest numerical method for solving SDEs. For an SDE of the form $dX_t = f(t, X_t)dt + g(t, X_t)dW_t$, the update rule is:
$$X_{n+1} = X_n + f(t_n, X_n)\Delta t + g(t_n, X_n)\Delta W_n$$
where $\Delta W_n \sim \mathcal{N}(0, \Delta t)$. In this codebase, it is used for basic trajectory rollouts and baseline comparisons.

### BAOAB Scheme (`BAOAB`)
The BAOAB scheme is a sophisticated operator-splitting method used particularly for Langevin dynamics to improve stability and accuracy, especially for larger timesteps $\Delta t$. It splits the SDE update into three parts:
- **B (Drift)**: A deterministic update (half-step) using the drift: $x \gets x + \frac{\Delta t}{2} f(t, x)$.
- **A (Position)**: An update to the positions based on current momenta.
- **O (Ornstein-Uhlenbeck)**: An update to the momenta incorporating both friction and random noise (the stochastic part).
- **A (Position)**: Another position update.
- **B (Drift)**: A final deterministic update (half-step) using the drift: $x \gets x + \frac{\Delta t}{2} f(t, x)$.

This "sandwich" structure (B-A-O-A-B) provides second-order accuracy for the position updates and is more robust than Euler-Maruyama when integrating the underdamped Langevin dynamics used in this project.

## 4. Boundary Conditions
Boundary conditions define the initial and final distributions of the trajectories. They are implemented in `src/kldescent/sde/boundary.py` and can be specified using the `--boundary` argument.

### UnderdampedLandauer
This class implements initial and target distributions modelling Landauer's bit erasure.
- **Initial Distribution**: $p_0(q, p) = p_0(q) \mathcal{N}(p; 0, I)$, where $p_0(q)$ is the initial position marginal.
- **Target Distribution**: $p_T(q, p) = p_T(q) \mathcal{N}(p; 0, I)$, where $p_T(q)$ is the target position marginal.

The position marginals are given by:
$$p_0(q) = \frac{1}{Z_0} e^{-U_{\text{initial}}(q)}, \quad p_T(q) = \frac{1}{Z_T} e^{-U_{\text{final}}(q)}$$
where $U_{\text{initial}}$ and $U_{\text{final}}$ are the Landauer potentials. The drift is trained to bridge these two distributions.

### UnderdampedGaussian
Both the initial and target distributions are Gaussians.
- **Initial Distribution**: $\mathcal{N}(\mu_{\text{initial}}, \sigma_{\text{initial}}^2)$
- **Target Distribution**: $\mathcal{N}(\mu_{\text{final}}, \sigma_{\text{final}}^2)$

The parameters ($\mu_{\text{initial}}, \sigma_{\text{initial}}, \mu_{\text{final}}, \sigma_{\text{final}}$) are passed via `gaussian_params()` (which typically reads from command-line arguments or config files). This is useful for testing the bridge from one Gaussian state to another in an underdamped system.


## 5. Training Process

### Training Loop
The training is managed by `Train` in `src/kldescent/model/train.py` and orchestrated by `train_loop_du` in `src/kldescent/trainloop.py`.

1. **Trajectory Rollout**: For each training step, trajectories are integrated from the initial boundary conditions (specified by `--boundary`) using a specified integrator (e.g., Euler-Maruyama or BAOAB).
2. **Gradient Computation**: The loss $\hat{L}$ is calculated, and gradients are backpropagated through the SDE trajectories to the neural network weights.
3. **Parameter Updates**:
    - Two separate `AdamW` optimizers are used: one for the **Fourier Features** and one for the **Feed-Forward Layers**.
    - **Learning Rate**: The initial learning rate is set by `--lr` (default: `0.0005`).
    - **Gradient Clipping**: Applied to prevent exploding gradients, with the threshold set by `--gradclip` (default: `0.5`).
4. **Learning Rate Scheduling**: 
    - `CosineAnnealingLR` is used to decay the learning rate over time.
    - The schedule starts decaying after a specific number of epochs defined by `--startdecay` (default: `20`).

### General Configuration
- **Epochs**: The total number of training epochs is set by `--epochs` (default: `1`).
- **Logging**: Checkpoints and metrics are saved every `--logstep` epochs (default: `1`) into the folder specified by `--folder` (default: `results2`).
- **Device**: Computing device is specified by `--device` (default: `cpu`).

### Pre-training
If pre-training data is provided (via `--pretrain`), the network is initialized using data from a CSV file before the KL-descent process begins.

## 5. Key Components Summary

| Component | Location | Responsibility |
| :--- | :--- | :--- |
| `NeuralNetwork` | `src/kldescent/model/nn.py` | Parametrizes the drift $du(t, x)$. |
| `kl` (Loss) | `src/kldescent/sde/loss.py` | Computes the KL-divergence based loss via SDE rollout. |
| `Train` | `src/kldescent/model/train.py` | Handles optimizers, gradient clipping, and the `train_step`. |
| `train_loop_du` | `src/kldescent/trainloop.py` | Manages epochs, scheduler steps, and checkpointing. |
| `TerminalCost` | `src/kldescent/sde/terminalcost.py` | Computes the penalty $g$ for the final state distribution. |

