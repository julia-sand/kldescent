import numpy as np
import torch
from model import *
from sde import *
from utils import *
import viz.plots as plots
import pdb
import os

#parse arguments
args = parser.overdamped_argparser()

#initialise the model
#TODO: add gpu option, decide architectures
#TODO: add config file to read these params
if args.boundary=="Landauer":
    boundary = boundary.HalfBridgeLandauer()
    net = nn.NeuralNetwork(args.hiddenlayerneurons,
                            args.tsteps,
                            boundary,
                            args.checkpointpath).to(args.device)

else: #Linear OU
    boundary = boundary.LinearOU(args.Tf)
    net = nn.MLP(args.hiddenlayerneurons,args.checkpointpath).to(args.device)

#create a log file and results directory
plotter = plots.PlotFuns(boundary)
logger = logger.LossLogger(plotter)
plotter.setresultsdir(logger.get_results_path())
#intialise plotter

#pretraining - train to approximate a numeric drift given by a csv file
if args.pretrain:
    import run_pretrain
    run_pretrain.run_pretrain(net,logger.get_results_path(),boundary)

#intialise sde integration 
integrator = langevin.SDEOverdamped(args.batchsamples,
                                    args.epsilon,
                                    args.Tf,
                                    args.tsteps,
                                    boundary,
                                    logger)

#plot intiial values
integrator.simulation_free(net,True,0)

#initialise trainer 
trainer = train.Train(integrator, net, args.logstep) 

for epoch in range(1,args.epochs+1):
    trainer.train_step(epoch)