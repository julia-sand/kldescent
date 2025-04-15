import numpy as np
import torch
import model.model as model
from utils import *
import viz.plots as plots
import time

###===Parameters===###
# #TODO: add these as command line options (argparse)
epsilon = 1
Tf = 0.2 #final time
tstep = 0.005 #size of time step in mesh
epochs = 1 #number of times to generate data and update the params. 
logstep = 10 #how often to log the loss and make plots and checkpoints.
batch_samples = 2000 # number of samples in each step 
checkpoint_path = "results/run20250415-124112/checkpoints/net_checkpoint_0.pt" #None #"kldescent/tests/net_pretrained.pt" add path to torch checkpoint 
pre_train = True #whether or not to pretrain the network (#TODO this is implemented only for dummy data- does not work for perturbative.)
path_to_pretrain = "src/refsolution/value.csv" #csv of the data to use for pretraining

##make a directory for the results
import os 
dirname = os.path.dirname(__file__) #fetches current location 
run_id = time.strftime("%Y%m%d-%H%M%S")

newpath = os.path.join(dirname,f'../results/run{run_id}')
os.makedirs(newpath)
os.makedirs(newpath+"/checkpoints") #add checkpoints folder
    


#create a log file
logger = logger.Logger(newpath +r'/loss.csv')
#intialise plots
plotter = plots.PlotFuns(newpath)

device = "cpu"

#time mesh
#TODO: time encoding ?
t_vec = np.linspace(0,Tf,int(np.floor(Tf/tstep)+1))

#initialise the model
#TODO: add gpu option, decide architectures
net = model.NeuralNetwork().to(device)

##option: load saved model 
if checkpoint_path is not None:
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print("Checkpoint loaded successfully.")


#pretraining - train to approximate a numeric drift given by a csv file
if pre_train:

    #save checkpoint so we have a record of the intialised params (or resaved checkpoint)
    torch.save(net.state_dict(),newpath+"/checkpoints/initialisation.pt") 

    print("Starting pre-training")
    import pretrain
    pretrain.pretraining_perturb(net,path = path_to_pretrain, n_epochs = 30000)
    print("Pretraining completed successfully.")

    os.makedirs(newpath+"/pretrain")
    #TODO : call plotters here. 
    #plotter.setsubdir("/pretrain")
    
    #save pretrained parameters into checkpoint
    #torch.save(net.state_dict(),newpath+"/pretrain/pretrained_checkpoint.pt") 
    

#copy params into fixed network 
net_fixed = model.NeuralNetwork().to(device)
net_fixed.load_state_dict(net.state_dict())
#net_fixed.parameters(requires_grad=False) #remove any gradients. 
net_fixed.eval() #put into evaluation mode

#TODO: how often should we log the results? 
saved_epochs = list(range(0,epochs,logstep))

#set up optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
net.train() #set to train mode

for epoch in range(epochs):

    #reset gradients 
    optimizer.zero_grad()

    #generate initial training data
    #in this testing example, we use a point mass at x=0. See Section 2.5
    xinit = torch.zeros((batch_samples,1))

    #evolve the sde 
    Lhat,xout = sde.simulation_free_overdamped(xinit,net,net_fixed,t_vec,batch_samples,epsilon)
    if (epoch in saved_epochs) or (epoch == epochs -1):
        #make new folder for this epoch 
        os.makedirs(newpath+f"/epoch{epoch}") #add epoch plots folder
        plotter.setsubdir(f"/epoch{epoch}")
        print(plotter.subdir)
        #make plots
        plotter.marginal_distribution(xout.detach().cpu().numpy())
        
        plot_times = [0,0.1,0.2] #these will be plotted
        dus = []
        for t in plot_times:
                duplot = net(torch.column_stack((t*torch.ones(batch_samples,dtype=torch.float32)
                                                    ,torch.linspace(-3,3,batch_samples)))).detach().cpu().numpy()
                duplot_fixed = net_fixed(torch.column_stack((t*torch.ones(batch_samples,dtype=torch.float32)
                                                    ,torch.linspace(-3,3,batch_samples)))).detach().cpu().numpy()
                dus.append([duplot,duplot_fixed])
                
        plotter.plot_drift(np.linspace(-3,3,batch_samples),times = plot_times,dus=dus)
        
        #get kl divergence from assigned final
        kl_val = kl.kl_div(xout.detach().cpu().numpy(),sampler.generate_samples_q(batch_samples,mode="final"))

        logger.log(epoch,Lhat.mean().item(),kl_val) #log the loss 


        #save checkpoint
        torch.save(net.state_dict(),newpath+f"/checkpoints/net_checkpoint_{epoch}.pt") 
    
    ##training##
    #Lhat.mean().backward()
    #optimizer.step()
    
    #copy updated params into fixed network 
    net_fixed.load_state_dict(net.state_dict())
    #TODO: can this network be replaced simply with "no grad" for the theta bars?

    