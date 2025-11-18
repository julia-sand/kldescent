import os

import numpy as np
import torch
from torch.nn import DataParallel

#import pdb
#from memory_profiler import profile

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
from src.utils import *
import src.viz.plots as plots
from src.viz.checkpoint_plots import *

#@profile
def run():

    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()
    metrics_logger.log_dict(args)

    #initialise the model
    boundaryconditions = getattr(boundaries,args["boundary"])(logger=metrics_logger)
    if args["boundary"]=="LinearOU":
        net = nn.MLP().to(args["device"])
    else:
        net = nn.NeuralNetwork(boundaryconditions).to(args["device"])

    metrics_logger.log_dict(net.params)
    net = DataParallel(net)

    if args["pretrain"]: #TODO this needs updating
        from model.pretrain import run_pretrain
        run_pretrain.run_pretrain(net,metrics_logger.get_results_path(),boundaryconditions)
    
    #intialise sde integration 
    integrator = langevin.SDE(boundaryconditions)
    metrics_logger.log_dict(integrator.sde_params)

    #initialise trainer & ref solution computation
    trainer = train.Train(net,integrator) 
    metrics = train.Metrics(None, net, integrator, args["logstep"],metrics_logger)

    for epoch in range(1,args["epochs"]+1):
        try:
            loss_out = trainer.train_step()
            metrics.update_metrics(epoch, loss_out)
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {epoch}")
        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
        except slurmhandler.TimeLimitException: #TODO check this works?
            print(f"The training must be stopped because SLURM is about to kill the job. Training ended at epoch {epoch}")
            metrics_logger.save_checkpoint(net,epoch)


    plot_utils = PlotUtils(metrics_logger.get_results_path(),epoch=args["epochs"])

    plot_utils.checkpoint_plots_du()

if __name__=="__main__":
    run()
