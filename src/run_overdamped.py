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
    metrics_logger.log_args(args)
    metrics_logger.log_dict(args)

    #initialise the model
    #TODO: add gpu option, decide architectures
    #TODO: add config file to read these params
    #dynamically create boundary class bsaed on input
    boundaryconditions = getattr(boundaries,args.boundary)()
    if args.boundary=="LinearOU":
        net = nn.MLP().to(args.device)
    else:
        net = nn.NeuralNetwork(boundaryconditions).to(args.device)

    #save all params
    metrics_logger.log_dict(boundaryconditions.params)
    metrics_logger.log_dict(net.params)
    
    net = DataParallel(net)

    try:
        metrics_logger.log_dict(boundaryconditions.gauss_boundary)
    except AttributeError: #skip if not available
        pass

    if args.pretrain:
        import run_pretrain
        run_pretrain.run_pretrain(net,metrics_logger.get_results_path(),boundaryconditions)

    
    
    #intialise sde integration 
    #TODO add these arguments so they get read directly from the command line already in the sde file
    integrator = langevin.SDEOverdamped(boundaryconditions,
                                        metrics_logger)

    #initialise trainer 
    trainer = train.Train(net,integrator) 
    metrics = train.Metrics(trainer,args.logstep)

    for epoch in range(0,args.epochs+1):
        try:
            loss_out = trainer.train_step()
            metrics.update_metrics(epoch,loss_out)
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {epoch}")
            #plot the loss metrics
            #plotter.plot_loss_metrics()
        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
        except slurmhandler.TimeLimitException:
            print(f"The training must be stopped because SLURM is about to kill the job. Training ended at epoch {epoch}")
            metrics_logger.save_checkpoint(net,epoch)
            #plotter.plot_loss_metrics()


    plotter = plots.PlotFuns(boundaryconditions)
    plotter.setresultsdir(metrics_logger.get_results_path())
    checkpoint_plots_all(metrics_logger,
                            boundaryconditions,
                            net,
                            plotter,
                            integrator)
    plotter.plot_loss_metrics()

if __name__=="__main__":
    run()
