"""Train the NN from a YAML file of command-line options and print the final position variance.

    PYTHONPATH=src python train_from_yaml.py params.yaml

Every key in the YAML is passed to the training CLI as `--key value`. The position variance is computed with the
library: integrator.update_ref_statistics() (samples at Tf under the trained drift) + integrator.compute_cumulant_row().
"""
import glob, re, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
cfg.setdefault("boundary", "UnderdampedGaussian")
cfg.setdefault("folder", "results_yaml")
sys.argv = [sys.argv[0]] + [t for k, v in cfg.items() if v is not None for t in (f"--{k}", *map(str, v if isinstance(v, list) else [v]))]

import torch
import kldescent.run
import kldescent.sde.boundary as boundaries
from kldescent.model import nn
from kldescent.sde import langevin
from kldescent.utils import parser

kldescent.run.run()                                              # training, exactly as `python -m kldescent.run`

args = parser.overdamped_argparser()
boundary = getattr(boundaries, args["boundary"])()
net = nn.NeuralNetwork(boundary, args["device"])
ck = max(glob.glob(f"{cfg['folder']}/checkpoints/du_checkpoint_*.pt"), key=lambda f: int(re.findall(r"_(\d+)\.pt$", f)[0]))
net.load_state_dict(torch.load(ck, weights_only=True, map_location=args["device"]))
integrator = getattr(langevin, args["integrator"])(boundary, net=net)

xT = integrator.update_ref_statistics(testing_samples=cfg.get("nsamples", 50000))
mean_q, mean_p, var_q, var_p, cross = integrator.compute_cumulant_row(xT)[0].tolist()
print(f"checkpoint {ck}")
print(f"position variance at Tf = {var_q:.5f}")