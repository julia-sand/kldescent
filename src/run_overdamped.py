import numpy as np
import torch
from model import *
from sde import *
from utils import *
import viz.plots as plots
import pdb
import run_pretrain

#create a log file
logger = logger.LossLogger()
#intialise plots
plotter = plots.PlotFuns(logger.get_results_path())
#parse arguments
args = parser.overdamped_argparser()
#initalise boundary data
boundary = boundary.AssignedBoundaryData()

#initialise the model
#TODO: add gpu option, decide architectures
net = nn.NeuralNetwork().to(args.device)

##option: load saved model 
if args.checkpointpath is not None:
    net.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
    print("Checkpoint loaded successfully.")

#pretraining - train to approximate a numeric drift given by a csv file
if args.pre_train:
    
    run_pretrain.run_pretrain(net,logger.get_results_path())



#copy params into fixed network 
net_fixed = nn.NeuralNetwork().to(args.device)
net_fixed.load_state_dict(net.state_dict())
#net_fixed.parameters(requires_grad=False) #remove any gradients. 
net_fixed.eval() #put into evaluation mode


#set up optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
net.train() #set to train mode

#intialise sde integration and utility functions
integrator = langevin.SDEOverdamped(args.batch_samples,args.epsilon,args.Tf,args.t_steps)
for epoch in range(args.epochs):

    #reset gradients 
    optimizer.zero_grad()

    #evolve the sde 
    Lhat,xout = integrator.simulation_free(net,net_fixed)
    
    if (epoch % args.logstep == 0) or (epoch == args.epochs-1):
        logger.log_epoch_data(plotter,net,net_fixed,epoch,Lhat,xout)

    ##training##
    Lhat.mean().backward()
    optimizer.step()
    
    #copy updated params into fixed network 
    net_fixed.load_state_dict(net.state_dict())
    #TODO: can this network be replaced simply with "no grad" for the theta bars?
