import numpy as np
import torch
import model
import pretraining 
import matplotlib.pyplot as plt
from utils.sampler import generate_samples, generate_samples_q
import sde 
import viz.plots as plots
from utils.logger import Logger
import kl 
##TODO: merge this file with run.py and add overdamped/underdamped as parsable argument. 

#create a log file
logger = Logger(r'kldescent/results/run1/loss.csv')

###
#TODO: add these as command line options (argparse)
batch_samples = 10 # number of samples in each step 
checkpoint_path = None #"kldescent/tests/net_pretrained.pt" add path to torch checkpoint 
pre_train = False #whether or not to pretrain the network (#TODO this is implemented only for dummy data- does not work for perturbative.)



device = "cpu"

#initialise the model
#TODO: add gpu option, decide architecture
net1 = model.NeuralNetwork().to(device)
net_fixed = model.NeuralNetwork().to(device)

#pretraining - train to approximate the perturbative 
#uncomment to initialise the neural network by training it on dummy data
if pre_train:
    pretraining.pretraining_perturb(net1)

##option: load saved model 
if checkpoint_path is not None:
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))

#copy params into fixed network 
net_fixed.load_state_dict(net1.state_dict())

t_vec = np.linspace(0,0.2,11)
epsilon = 0.2
epochs = 2 #number of times to generate data and update the params. 

#TODO: how often should we log the results? 
saved_epochs = [1,2,3]

for epoch in range(epochs):
    net1.train() #set to train mode

    #generate initial training data
    #in this testing example, we use a point mass at x=0. See Section 2.5
    xinit = torch.zeros((batch_samples,1))
    #generate_samples_q(batch_samples,mode="initial")

    #evolve the sde 
    xout = sde.simulation_free_overdamped(xinit,net1,net_fixed,t_vec,batch_samples,epsilon,logger)
    

    ##GRADIENT DESCENT STEP GOES HERE!!!

    if epoch in saved_epochs:
        #make plots
        plots.PlotFuns().marginal_distribution(xout.detach().cpu().numpy(),epoch)

        #save checkpoint
        torch.save(net1.state_dict(),f"kldescent/tests/net_checkpoint_{epoch}.pt")

