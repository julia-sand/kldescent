# Benchmarks

The code also computes benchmark solutions for comparison with the numerics. 

To compute the benchmark solution, set a folder to `--usebenchmark` to save the results. The type of benchmark solution depends on the boundary conditions given. If benchmark is requested, the training will use as a comparison in the metrics, and include benchmark values in the plots.

Benchmark results are saved as CSV files, so they can be reused once computed.

### Gaussian BVP (`gaussian.py`)

The benchmark is the solution of a first-order system of ordinary differential equations. `GaussBVP` solves the two-point boundary value problem (BVP) exactly with `scipy.integrate.solve_bvp`. The BVP is stiff for large values of $\varepsilon$, so the solver uses numerical continuation. This means starting from a small $\varepsilon$, stepping up to the target $\varepsilon$ and using the previous solution as the initial guess for the next. If a step that fails to converge, the step is bisected and retried until a maximum number of tries. The initialisation is the free diffusion (i.e. $U=0$).

### Perturbative solutions (`perturbative/`, `overdamped/`)

For non-Gaussian boundary conditions, we use the perturbative expansion described in the analytic portion of the paper. This is done in two steps

1. **Overdamped cell problem** (`overdamped/computesol.py`, `overdamped/iteration.py`): the overdamped cell problem is solved by iterating two coupled heat equations between the initial and target marginals, found by a Hopf-Cole transform of the partial differential equations specifying the first order optimality conditions. The results are transformed back to give the overdamped value function, probability density and drift on a $(t, q)$ grid, which is saved to `overdamped.csv`.
2. **Perturbative corrections** (`perturbative/corrections.py`): the analytic corrections are computed to give predictions for the underdamped the drift and distribution at $O(\varepsilon)$. The results are saved to `perturbative_underdamped.csv` and the predicted cumulants are saved to `perturbative_cumulants.csv`.

If any required files (`overdamped.csv`, `perturbative_underdamped.csv`, or the drift checkpoint) are missing from the `--usebenchmark` folder, the missing data is recomputed automatically when the benchmark is initialised.
