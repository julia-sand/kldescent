import numpy as np
import torch
from model import *
from sde import *
from utils import *
import viz.plots as plots
import time
import pdb

###===Parameters===###
# #TODO: add these as command line options (argparse)

##make a directory for the results
import os 
dirname = os.path.dirname(__file__) #fetches current location 
run_id = time.strftime("%Y%m%d-%H%M%S")

newpath = os.path.join(dirname,f'../results/{run_id}')
os.makedirs(newpath)
os.makedirs(newpath+"/checkpoints") #add checkpoints folder


#create a log file
logger = logger.Logger(newpath +r'/loss.csv')
#intialise plots
plotter = plots.PlotFuns(newpath)
#parse arguments
parsedargs = parser.overdamped_argparser()


#initialise the model
#TODO: add gpu option, decide architectures
net = nn.NeuralNetwork().to(parsedargs["device"])

##option: load saved model 
if parsedargs["checkpointpath"] is not None:
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print("Checkpoint loaded successfully.")

#pretraining - train to approximate a numeric drift given by a csv file
if parsedargs["pre_train"]:

    #save checkpoint so we have a record of the intialised params (or resaved checkpoint)
    torch.save(net.state_dict(),newpath+"/checkpoints/initialisation.pt") 

    print("Starting pre-training")
    
    try:
        train.Pretrain(parsedargs["path_to_pretrain"]).pretraining_perturb(net)
        print("Pretraining completed successfully.")

    # save if job is manually interrupted
    except KeyboardInterrupt:
        torch.save(net.state_dict(),newpath+"/checkpoints/current_params.pt") 
        print("Pretraining interrupted. Current parameters saved.")


#copy params into fixed network 
net_fixed = nn.NeuralNetwork().to(parsedargs["device"])
net_fixed.load_state_dict(net.state_dict())
#net_fixed.parameters(requires_grad=False) #remove any gradients. 
net_fixed.eval() #put into evaluation mode


#set up optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
net.train() #set to train mode

#intialise sde integration and utility functions
integrator = sde.SDEOverdamped(parsedargs["batch_samples"],parsedargs["epsilon"],parsedargs["Tf"],parsedargs["t_steps"])
for epoch in range(epochs):

    #reset gradients 
    optimizer.zero_grad()

    #evolve the sde 
    Lhat,xout = integrator.simulation_free(net,net_fixed)
    
    if (epoch % logstep == 0) or (epoch == epochs -1):
        
        #make new folder for this epoch 
        os.makedirs(newpath+f"/epoch{epoch}") #add epoch plots folder
        plotter.setsubdir(f"/epoch{epoch}")

        plotter.plot_drift(np.linspace(-4,4,batch_samples),times = plot_times,dus = integrator.get_net_output(net,net_fixed,plot_times))

        #get kl divergence from assigned final
        kl_val = kl.kl_div(xout.detach().cpu().numpy(),sampler.generate_samples_q(batch_samples,mode="final"))
        logger.log(epoch,Lhat.mean().item(),kl_val) #log

        #make plots
        plotter.marginal_distribution(xout.detach().cpu().numpy())

        #save checkpoint
        torch.save(net.state_dict(),newpath+f"/checkpoints/net_checkpoint_{epoch}.pt") 
    
    ##training##
    Lhat.mean().backward()
    optimizer.step()
    
    #copy updated params into fixed network 
    net_fixed.load_state_dict(net.state_dict())
    #TODO: can this network be replaced simply with "no grad" for the theta bars?

    