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
boundary = boundary.AssignedBoundaryData()
#create a log file and results directory
logger = logger.LossLogger()
#intialise plotter
plotter = plots.PlotFuns(logger.get_results_path(),boundary)

#initialise the model
#TODO: add gpu option, decide architectures
net = nn.NeuralNetwork(args.checkpointpath).to(args.device)

#pretraining - train to approximate a numeric drift given by a csv file
if args.pretrain:
    import run_pretrain
    run_pretrain.run_pretrain(net,logger.get_results_path())

#set up optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.train() #set to train mode

#intialise sde integration 
integrator = langevin.SDEOverdamped(args.batch_samples,args.epsilon,args.Tf,args.t_steps,boundary)
#plot intiial values

Lhat,xout = integrator.simulation_free(net)
logger.log_epoch_data(plotter,net,0,Lhat,0,xout.detach().cpu().numpy())

for epoch in range(1,args.epochs+1):

    Lhat,xout = integrator.simulation_free(net)

    if (epoch % args.logstep == 0):
        try: #since the training fails a lot of the time, keep this to continue debug mode and not error. 
            kl_val = kl.kl_div(xout.detach().cpu().numpy(),boundary.generate_samples_q(args.batch_samples,mode="final"))
        except:
            kl_val = 0
        #evolve the sde 
        logger.log_epoch_data(plotter,net,epoch,Lhat,kl_val,xout.detach().cpu().numpy())

    #reset gradients 
    optimizer.zero_grad()

    ##training##
    lhat_out = Lhat.mean()
    lhat_out.backward()
    optimizer.step()

    #copy updated params into fixed network 
    #TODO: can this network be replaced simply with "no grad" for the theta bars?
