import numpy as np
import torch
import model
import pretraining 
import matplotlib.pyplot as plt
from utils.sampler import generate_samples
import sde 
import viz.plots as plots
import utils.logger as logger
import kl 


###
#TODO: add these as command line options (argparse)
batch_samples = 1000 # number of samples in each step 
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

#copy params into fixed network 
net_fixed.load_state_dict(net1.state_dict())


##option: load saved model 
if checkpoint_path is not None:
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))

t_vec = np.linspace(0,0.2,11)
epsilon = 0.2
epochs = 2 #number of times to generate data and update the params. 

#TODO: how often should we log the results? 
saved_epochs = [1,2,3]

##set up optimiser 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(epochs):
    net1.train() #set to train mode

    #generate initial training data
    xinit = generate_samples(batch_samples,mode="initial")

    #evolve the sde 
    Lhat = sde.sde_underdamped(xinit,net1,net_fixed,t_vec,batch_samples,epsilon)
    
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updateing the parameters after each iteration
    optimizer.step()
    #log the loss function metrics
    logger.log([epoch,losscurr.item(),kl.kl_div(xout)])
    
    ##GRADIENT DESCENT STEP GOES HERE!!!

    if epoch in saved_epochs:
        #make plots
        plots.PlotFuns().joint_distribution_scatter(xout.detach().cpu().numpy(),epoch)

        #save checkpoint
        torch.save(net1.state_dict(),f"kldescent/tests/net_checkpoint_{epoch}.pt")
    
    # zeroing gradients after each iteration
    optimizer.zero_grad()