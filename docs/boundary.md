# Boundary Conditions

The boundary conditions refers to the assigned initial and target distributions. These are defined in the file `src/kldescent/sde/boundary.py`. To set the boundary conditions, use `--boundary`.

### UnderdampedLandauer

The assigned initial and target distributions modelling Landauer's bit erasure. 
- **Initial Distribution**:
This is a single peaked distribution. 
$$p_\iota(q, p) \propto e^{-(q - 1)^4} e^{\frac{1}{2}p^2}$$
- **Target Distribution**:
The target is a double-peaked distribution.
$$p_f(q, p) \propto e^{-(q^2 - 1)^2} e^{\frac{1}{2}p^2}$$

### UnderdampedGaussian

Both the initial and target distributions are Gaussians:
- **Initial Distribution**:
$$p_\iota(q, p) \propto e^{-\frac{1}{2\sigma_\iota^2}(q-\mu_\iota)^2} e^{\frac{1}{2}p^2}$$
- **Target Distribution**:
$$p_f(q, p) \propto e^{-\frac{1}{2\sigma_f^2}(q-\mu_f)^2} e^{\frac{1}{2}p^2}$$

The parameters, $\mu_\iota,\,\mu_f,\,\sigma_\iota,\,\sigma_f$ are passed via `gaussian_params()` with `--muinitial`, `--mufinal`, `--sigmainitial`, `--sigmafinal` respectively.
