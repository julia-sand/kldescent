import numpy as np
import torch
import model
import pretraining 
from utils.sampler import generate_samples, generate_samples_q
import sde 
import viz.plots as plots
from utils.logger import Logger
import kl 
##TODO: merge this file with run.py and add overdamped/underdamped as parsable argument. 

#create a log file
logger = Logger(r'kldescent/results/run1/loss.csv')

#initialise plots
plotter = plots.PlotFuns()

###
#TODO: add these as command line options (argparse)
batch_samples = 200 # number of samples in each step 
checkpoint_path = "kldescent/results/run1/checkpoints/net_checkpoint_190.pt" #None #"kldescent/tests/net_pretrained.pt" add path to torch checkpoint 
pre_train = False #whether or not to pretrain the network (#TODO this is implemented only for dummy data- does not work for perturbative.)



device = "cpu"

#initialise the model
#TODO: add gpu option, decide architecture
net = model.NeuralNetwork().to(device)
net_fixed = model.NeuralNetwork().to(device)

#pretraining - train to approximate the perturbative 
#uncomment to initialise the neural network by training it on dummy data
if pre_train:
    pretraining.pretraining_perturb(net)

##option: load saved model 
if checkpoint_path is not None:
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))

#copy params into fixed network 
net_fixed.load_state_dict(net.state_dict())

t_vec = np.linspace(0,0.2,11)
epsilon = 0.2
epochs = 1#number of times to generate data and update the params. 

#TODO: how often should we log the results? 
saved_epochs = list(range(0,epochs+10,10))

#set up optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
net.train() #set to train mode

for epoch in range(epochs):

    #generate initial training data
    #in this testing example, we use a point mass at x=0. See Section 2.5
    xinit = torch.zeros((batch_samples,1))
    #generate_samples_q(batch_samples,mode="initial")

    #evolve the sde 
    Lhat,xout = sde.simulation_free_overdamped(xinit,net,net_fixed,t_vec,batch_samples,epsilon)
    Lhat.mean().backward()
    optimizer.step()
    print(net[0].weight.grad) 

    if epoch in saved_epochs:
        #make plots
        plotter.marginal_distribution(xout.detach().cpu().numpy(),epoch)
        
        #get kl divergence from assigned final
        #
        kl_val = kl_div(xout.detach().cpu().numpy(),generate_samples_q(batch_samples,mode="final"))

        logger.log(epoch,Lhat.mean().item(),kl_val) #log the loss 

        #save checkpoint
        torch.save(net.state_dict(),f"kldescent/results/run1/checkpoints/net_checkpoint_{epoch}.pt")

