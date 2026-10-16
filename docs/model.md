# Model Specification

This file describes the architecture neural network used to parametrise the control. 

## 1. Model Architecture

The drift is parameterized by a neural network model, described in the `NeuralNetwork` class (`src/kldescent/model/nn.py`). 

### Input Normalisation

To standardise the inputs time $t$ and position $q$ to be in the same scale, we define normalisation in `MinMaxNorm_layers`.
- Time:
$$\hat{t} = \dfrac{t}{T_f}$$
- Position:
$$\hat{q} = \dfrac{q + |\text{xmax}|}{2 |\text{xmax}| + 10^{-8}}$$
where `xmax` is `boundary.norm_qmax` (e.g. 3.5 for `UnderdampedLandauer`, 5 for `UnderdampedGaussian`).

### Network Branches

The neural network has three branches to transform its inputs into higher dimensions.
1.  **Time Branch (`NNt`)**: a feed-forward network
$$\text{NN}_t : \hat{t} \to \text{Linear} \to \text{GELU} \to \text{Linear} \to \text{GELU} \to \mathbb{R}^H.$$
2.  **State Branch (`NNx`)**: a feed-forward network
$$\text{NN}_x : \hat{x} \to \text{Linear} \to \text{GELU} \to \text{Linear} \to \mathbb{R}^H.$$
3.  **Fourier Branch (`FourierFeatures`)** [3]: a high-frequency encoding of time
$$F(\hat{t}) = \bigl[\sin(2\pi B\hat{t}), \cos(2\pi B\hat{t})\bigr] \in \mathbb{R}^{2H},$$
where $B \in \mathbb{R}^{H \times 1}$ is a linear layer without bias, initialised $\mathcal{N}(0, \text{scale}^2)$. The scale is 
$$\text{scale} = 0.3 \, T_f \sqrt{1 + 4\varepsilon^2}.$$

Let $H$ denote the hidden width of the network, set by the command-line argument `--duhiddenlayerneurons` (default $16$). The number of neurons in each layer is:

| Layer | Inputs $\to$ outputs | Neurons in layer |
|---|---|---|
| Time branch `NNt`, 1st layer | $1 \to H$ | $H$ |
| Time branch `NNt`, 2nd layer | $H \to H$ | $H$ |
| State branch `NNx`, 1st layer | $d \to H$ | $H$ |
| State branch `NNx`, 2nd layer | $H \to H$ | $H$ |
| Fourier branch `FourierFeatures` ($B$) | $1 \to H$ | $H$ |
| `LayerNorm` | $2H \to 2H$ | — |
| Residual block `NN1`/`NN2`, $L_1$ | $2H \to H$ | $H$ |
| Residual block `NN1`/`NN2`, $L_2$ | $H \to H$ | $H$ |
| Residual block `NN1`/`NN2`, $L_3$ | $H \to 1$ | $1$ |


### Output

- **Combination**: the outputs of the time and state branches are concatenated and passed through a `LayerNorm`.[2]
- **Residual Blocks (`NN1`, `NN2`)**: the network uses two residual-like blocks (instances of the `NN2` module). Each block performs
$$\text{NN2}(v) = L_3\bigl(\text{GELU}(v_2 + \text{GELU}(L_2(v_2)))\bigr), \qquad v_2 = \text{GELU}(L_1(v)),$$
mapping $\mathbb{R}^{2H} \to \mathbb{R}^H \to \mathbb{R}^H \to \mathbb{R}^1$.
- **Final Output**: the resulting output $\mathbb{U}(t,q)$ is the sum of a learned component and term which is the target distribution's negative score (its gradient of the logarithm):
$$\boxed{\mathbb{U}(t,q) = \text{NN1}(LayerNorm([NNt(\hat{t}), NNx(\hat{q})])) + \text{NN2}\!\bigl(F(\hat{t})\bigr) \cdot \text{scorefinal}(q)}$$

## 2. Activation Function

Feed-forward branches and residual blocks use the GELU (Gaussian Error Linear Unit) activation function. This is implemented using PyTorch's `nn.GELU` [1]
$$\text{GELU}(x) = x\,\Phi(x)$$
where $\Phi$ is the cumulative distribution function of the standard normal distribution. 

## 3. References

[1] Hendrycks & Gimpel (2016), “Gaussian Error Linear Units (GELUs)”, arXiv:1606.08415.
[2] Ba, Kiros & Hinton (2016), “Layer Normalization”, arXiv:1607.06450
[3] Tancik et al. (2020), “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains”, arXiv:2006.10739