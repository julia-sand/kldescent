import numpy as np
import torch
from src.model import *
from src.sde import *
from src.utils import *
import src.viz.plots as plots
import pdb
import os

#parse arguments from command line
args = parser.overdamped_argparser()

#initialise the model
#TODO: add gpu option, decide architectures
#TODO: add config file to read these params
if args.boundary=="Landauer":
    boundary = boundary.HalfBridgeLandauer()
    net = nn.NeuralNetwork(args.hiddenlayerneurons,
                            boundary,
                            args.checkpointpath).to(args.device)
elif args.boundary=="LinearOU": #Linear OU
    boundary = boundary.LinearOU()
    net = nn.MLP(args.hiddenlayerneurons,args.checkpointpath).to(args.device)
elif args.boundary=="UnderdampedLandauer": #underdamped landauer
    boundary = boundary.UnderdampedHalfBridge()
    net = net = nn.NeuralNetwork(args.hiddenlayerneurons,
                            boundary,
                            args.checkpointpath).to(args.device)

#create a log file and results directory
plotter = plots.PlotFuns(boundary)
logger = logger.LossLogger(plotter)
plotter.setresultsdir(logger.get_results_path())

#pretraining - train to approximate a numeric drift given by a csv file
if args.pretrain:
    import run_pretrain
    run_pretrain.run_pretrain(net,logger.get_results_path(),boundary)

#intialise sde integration 
#TODO add these arguments so they get read directly from the command line already in the sde file
integrator = langevin.SDEOverdamped(args.batchsamples,
                                    boundary,
                                    logger)

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

plotter.plot_loss_metrics()