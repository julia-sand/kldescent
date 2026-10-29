"""Train the NN from a YAML config and compare its position variance over the WHOLE time interval [0, Tf]
with the benchmark solution. Prints the L2 difference

    L2 = sqrt( integral_0^Tf ( var_nn(t) - var_bench(t) )^2 dt )

    PYTHONPATH=src python test_variance_l2.py params.yaml

Config: same format as train_from_yaml.py (every key is passed to the training CLI as `--key value`;
`nsamples`, default 50000, is only used here: number of samples for the NN variance).

Benchmark: GaussBenchmarkSolution.solve_bvp_gaussian() for the boundary parameters in the config (half bridge),
position variance = get_perturbative_cumulant("Position Variance")  (t, c11).
NN: integrator.compute_sde_cumulants() on the benchmark's own time grid, position variance = column 2 of the cumulants
(the library's own function; its "reference" output is not used).
The integral is a trapezoid rule on the benchmark time grid.
"""
import contextlib, glob, io, os, re, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
cfg.setdefault("boundary", "UnderdampedGaussian")
cfg.setdefault("folder", "results_92")
sys.argv = [sys.argv[0]] + [t for k, v in cfg.items() if v is not None for t in (f"--{k}", *map(str, v if isinstance(v, list) else [v]))]

import numpy as np
import torch
import kldescent.run
import kldescent.sde.boundary as boundaries
from kldescent.benchmark.benchmarksol import GaussBenchmarkSolution
from kldescent.model import nn
from kldescent.sde import langevin
from kldescent.utils import parser
from kldescent.viz.checkpoint_plots import PlotUtils

kldescent.run.run()                                              # 1. training, exactly as `python -m kldescent.run`

args = parser.overdamped_argparser()
boundary = getattr(boundaries, args["boundary"])()
folder = cfg["folder"]

bench = GaussBenchmarkSolution(bridge_type="half", boundary=boundary, folder_name=f"{folder}/benchmark")
os.makedirs(f"{folder}/benchmark", exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):                  # 2. benchmark solution (the solver is chatty)
    bench.solve_bvp_gaussian()
t_b, var_bench = bench.get_perturbative_cumulant("Position Variance")

net = nn.NeuralNetwork(boundary, args["device"])                 # 3. NN position variance vs time
ck = max(glob.glob(f"{folder}/checkpoints/du_checkpoint_*.pt"), key=lambda f: int(re.findall(r"_(\d+)\.pt$", f)[0]))
epoch = int(re.findall(r"_(\d+)\.pt$", ck)[0])
net.load_state_dict(torch.load(ck, weights_only=True, map_location=args["device"]))

# Plot checkpoint plots at the last epoch
plot_utils = PlotUtils(epoch=epoch, benchmark=bench, folder=folder)
plot_utils.checkpoint_plots_du(usebenchmark=True, mcsamples=cfg.get("nsamples", 50000))

integrator = getattr(langevin, args["integrator"])(boundary, net=net)
torch.manual_seed(1234)
t_vec, _, _, cum_nn, _ = integrator.compute_sde_cumulants(benchmark=bench, mcsamples=cfg.get("nsamples", 50000))
var_nn = cum_nn[:, 2]                                            # [mean q, mean p, var q, var p, cross]
assert len(t_vec) == len(t_b) and np.allclose(t_vec, t_b, atol=1e-5), "NN and benchmark time grids differ"

diff = var_nn - var_bench                                        # 4. L2 difference over [0, Tf]
l2 = float(np.sqrt(np.trapezoid(diff ** 2, t_b)))
l2_bench = float(np.sqrt(np.trapezoid(var_bench ** 2, t_b)))
np.savetxt(f"{folder}/variance_vs_time.csv", np.column_stack([t_b, var_nn, var_bench]), delimiter=",",
           header="t,var_nn,var_bench", comments="")
print(f"checkpoint {ck}")
print(f"final position variance: nn {var_nn[-1]:.5f}   benchmark {var_bench[-1]:.5f}")
print(f"relative L2 difference = {l2 / l2_bench:.5f}   (L2 of the benchmark variance: {l2_bench:.5f})")
print(f"L2 difference of position variance over [0, {t_b[-1]:g}] = {l2:.5f}")