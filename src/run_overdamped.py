import numpy as np
import torch
from model import *
from sde import *
from utils import *
import viz.plots as plots
import pdb
import os
import torch.autograd as autograd
from torch.optim.lr_scheduler import CosineAnnealingLR


#parse arguments
args = parser.overdamped_argparser()

#initalise boundary data
if args.boundary=="Landauer":
    boundary = boundary.HalfBridgeLandauer()
else:
    boundary = boundary.LinearOU(args.Tf)

#create a log file and results directory
plotter = plots.PlotFuns(boundary)
logger = logger.LossLogger(plotter)
plotter.setresultsdir(logger.get_results_path())
#intialise plotter

#initialise the model
#TODO: add gpu option, decide architectures
net = nn.NeuralNetwork(args.hiddenlayerneurons,args.checkpointpath).to(args.device)

#pretraining - train to approximate a numeric drift given by a csv file
if args.pretrain:
    import run_pretrain
    run_pretrain.run_pretrain(net,logger.get_results_path(),boundary)

#set up optimizer
optimiser = torch.optim.Adam(net.parameters(), lr=0.003)
#COSINE ANNEALING LR SCHEDULE https://medium.com/@utkrisht14/cosine-learning-rate-schedulers-in-pytorch-486d8717d541
scheduler = CosineAnnealingLR(optimiser, T_max=100, eta_min=0.000001)
net.train() #set to train mode

#intialise sde integration 
integrator = langevin.SDEOverdamped(args.batchsamples,
                                    args.epsilon,
                                    args.Tf,
                                    args.tsteps,
                                    boundary,
                                    logger)

#plot intiial values
integrator.simulation_free(net,True,0)

for epoch in range(1,args.epochs+1):

    logflag = True if (epoch % args.logstep == 0) else False
    Lhat = integrator.simulation_free(net,logflag,epoch)

    #reset gradients 
    optimiser.zero_grad()

    ##training##
    Lhat.backward()
    optimiser.step()
    

    scheduler.step()
