import numpy as np
import torch
from model import *
from sde import *
from utils import *
import viz.plots as plots
import pdb
import os
import torch.autograd as autograd

#parse arguments
args = parser.overdamped_argparser()
#initalise boundary data
boundary = boundary.LinearOU(args.Tf)
#boundary.HalfBridgeLandauer()
#create a log file and results directory
plotter = plots.PlotFuns(boundary)
logger = logger.LossLogger(plotter)
plotter.setresultsdir(logger.get_results_path())
#intialise plotter

#initialise the model
#TODO: add gpu option, decide architectures
net = nn.NeuralNetwork(args.checkpointpath).to(args.device)

#pretraining - train to approximate a numeric drift given by a csv file
if args.pretrain:
    import run_pretrain
    run_pretrain.run_pretrain(net,logger.get_results_path(),boundary)

#set up optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
net.train() #set to train mode

#intialise sde integration 
integrator = langevin.SDEOverdamped(args.batch_samples,args.epsilon,args.Tf,args.t_steps,boundary)

#plot intiial values
integrator.simulation_free(net,True,logger)

for epoch in range(1,args.epochs+1):

    logflag=True if (epoch % args.logstep == 0) else False
    Lhat = integrator.simulation_free(net,epoch,logflag,logger)

    #reset gradients 
    optimizer.zero_grad()

    ##training##
    lhat_out = Lhat.mean()
    lhat_out.backward()
    optimizer.step()
