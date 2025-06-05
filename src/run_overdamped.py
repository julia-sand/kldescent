import os

import numpy as np
import torch

#import pdb
#from memory_profiler import profile

from src.model import *
from src.sde import *
from src.utils import *
import src.viz.plots as plots


#@profile
def run():

    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()
    metrics_logger.log_args(args)

    #initialise the model
    #TODO: add gpu option, decide architectures
    #TODO: add config file to read these params
    if args.boundary=="Landauer":
        boundaryconditions = boundary.HalfBridgeLandauer()
        net = nn.NeuralNetwork(boundaryconditions).to(args.device)
    elif args.boundary=="LinearOU": #Linear OU
        boundaryconditions = boundary.LinearOU()
        net = nn.MLP(args.hiddenlayerneurons,args.checkpointpath).to(args.device)
    elif args.boundary=="UnderdampedLandauer": #underdamped landauer
        boundaryconditions = boundary.UnderdampedHalfBridge()
        net = nn.NeuralNetwork(boundaryconditions).to(args.device)
    elif args.boundary=="UnderdampedNoMomentum": #
        boundaryconditions = boundary.UnderdampedNoMomentum()
        net = nn.NeuralNetwork(boundaryconditions).to(args.device)
    elif args.boundary=="UnderdampedHalfGaussian": #
        boundaryconditions = boundary.UnderdampedHalfGaussian()
        net = nn.NeuralNetwork(boundaryconditions).to(args.device)
        metrics_logger.log_args(boundaryconditions.gauss_boundary)
    elif args.boundary=="UnderdampedHalfGaussianNoMomentum": #
        boundaryconditions = boundary.UnderdampedHalfGaussianNoMomentum()
        net = nn.NeuralNetwork(boundaryconditions).to(args.device)
        metrics_logger.log_args(boundaryconditions.gauss_boundary)

    #plotter = plots.PlotFuns(boundaryconditions)
    #plotter.setresultsdir(metrics_logger.get_results_path())

    #save all params
    metrics_logger.log_args(boundaryconditions.params)
    metrics_logger.log_args(net.params)
    metrics_logger.log_dict(net.params)
    if args.pretrain:
        import run_pretrain
        run_pretrain.run_pretrain(net,metrics_logger.get_results_path(),boundaryconditions)

    #intialise sde integration 
    #TODO add these arguments so they get read directly from the command line already in the sde file
    integrator = langevin.SDEOverdamped(boundaryconditions,
                                        metrics_logger)

    #initialise trainer 
    trainer = train.Train(integrator, net, args.logstep) 

    for epoch in range(0,args.epochs+1):
        try:
            trainer.train_step(epoch)
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {epoch}")
            #plot the loss metrics
            #plotter.plot_loss_metrics()
        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
        except TimeLimitException:
            print(f"The training must be stopped because SLURM is about to kill the job. Training ended at epoch {epoch}")
            metrics_logger.save_checkpoint(net,epoch)
            #plotter.plot_loss_metrics()


    #plotter.plot_loss_metrics()

if __name__=="__main__":
    run()
