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
    overdampedlogger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()
    overdampedlogger.log_args(args)

    #initialise the model
    #TODO: add gpu option, decide architectures
    #TODO: add config file to read these params
    if args.boundary=="Landauer":
        boundaryconditions = boundary.HalfBridgeLandauer()
        net = nn.NeuralNetwork(args.hiddenlayerneurons,
                                boundaryconditions,
                                args.checkpointpath).to(args.device)
    elif args.boundary=="LinearOU": #Linear OU
        boundaryconditions = boundary.LinearOU()
        net = nn.MLP(args.hiddenlayerneurons,args.checkpointpath).to(args.device)
    elif args.boundary=="UnderdampedLandauer": #underdamped landauer
        boundaryconditions = boundary.UnderdampedHalfBridge()
        net = nn.NeuralNetwork(args.hiddenlayerneurons,
                                boundaryconditions,
                                args.checkpointpath).to(args.device)
    elif args.boundary=="UnderdampedNoMomentum": #
        boundaryconditions = boundary.UnderdampedNoMomentum()
        net = nn.NeuralNetwork(args.hiddenlayerneurons,
                                boundaryconditions,
                                args.checkpointpath).to(args.device)

    plotter = plots.PlotFuns(boundaryconditions)
    plotter.setresultsdir(overdampedlogger.get_results_path())

    #pretraining - train to approximate a n
    overdampedlogger.log_args(boundaryconditions.params)
    if args.pretrain:
        import run_pretrain
        run_pretrain.run_pretrain(net,overdampedlogger.get_results_path(),boundaryconditions)

    #intialise sde integration 
    #TODO add these arguments so they get read directly from the command line already in the sde file
    integrator = langevin.SDEOverdamped(boundaryconditions,
                                        overdampedlogger)

    #plot intiial values
    integrator.compute_metrics(net,0,0)

    #initialise trainer 
    trainer = train.Train(integrator, net, args.logstep) 

    for epoch in range(1,args.epochs+1):
        try:
            trainer.train_step(epoch)
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {epoch}")
            #plot the loss metrics
            plotter.plot_loss_metrics()
        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
    plotter.plot_loss_metrics()

if __name__=="__main__":
    run()
