import os

import numpy as np 
import torch
from torch import nn as torchnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR,ExponentialLR,ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

class Pretrain():
    """
    Class for training a neural network to interpolate CSV
    data to be useable as a function
    Used for eg. initialising a drift or training a secondary model to easier compute
    the perturbative evolutions
    """
    
    #define the training parameters and loop for any pretraining. 
    def __init__(self,refsol=None,path_to_pretrain=None):
        
        
        if path_to_pretrain is None:
            self.refsol = refsol 
            self.path = self.refsol.perturb_underdamped_loc #overdamped_loc        
        else:
            self.refsol = None  #overwrites the reference solution and learn straight from the csv
            self.path = path_to_pretrain 

        #set defaults of training process
        self.lr = 0.01 #initial learning rate
        self.gamma = 0.1 #decay rate of learning rate
        self.milestone = 5000 #frequency of learning rate decay (in epochs)
        self.n_epochs = 60000 #total number of epochs

        self.minibatch_samples = 512 #number of datarows to use per parameter update

        self.clip = 0.1 #size of gradient clipping

    def get_pretraining_data(self):
        """Load data from csv to be used in training the network""" 

        arr = np.loadtxt(self.path,
                        delimiter=",", skiprows=1)
        
        #remove zero probability areas
        #arr = arrtemp[np.where(arrtemp[:,2]>0)] 

        X = torch.tensor(arr[:,0:2], dtype=torch.float32)
        
        y = torch.tensor(arr[:,-1], dtype=torch.float32).reshape(-1, 1)
        
        return X,y

    def transform_data(self,X,y):
        r"""Implement a transformation for the data which can be customised depending on
        what the pre-training is doing. 
        
        This is based on the transformation
        (**) \mathbb{U}_t(q_t) = -\frac{\varepsilon}{2}\partial U_t(q_t)
        See the paper for more details.

        args:
            X : ndarray
                (t,q) coordinates 
            y : ndarray
                du(t,q)
        returns:
            X : ndarray
                same as input
            y : ndarray
                left hand side of (**) 
        """

        if self.refsol is not None:
            X[:,0] /= self.refsol.boundary.params["epsilon"]**2 #use t0 
            y *= (-(self.refsol.boundary.params["epsilon"]/(2**0.5)))

        return X,y

    def pretraining(self,net):   
        
        #set up training parameters
        loss_fn = torchnn.MSELoss()  
        optimizer = Adam(net.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".

        Xtemp,ytemp = self.get_pretraining_data()
        X,y = self.transform_data(Xtemp,ytemp)
 
        for epoch in tqdm(range(1,self.n_epochs+1)):
            
            #generate random batch 
            idx = torch.randint(low=0,high=X.shape[0],size=(self.minibatch_samples,))

            optimizer.zero_grad()
            y_pred = net(X[idx,:])
            loss = loss_fn(y_pred,y[idx])
            clip_grad_norm_(net.parameters(), self.clip)
            loss.backward()
            optimizer.step()       

            if epoch % self.milestone == 0:
                scheduler.step()
                #print(f'Finished epoch {epoch}, loss {loss}')
        
class Train():
    """
    Defines the parameters of the training process and the training steps
    for the neural network learning the non-perturbative drift.
    """

    def __init__(self, net, integrator):

        #set defaults of training process
        self.lr = 0.005 #initial learning rate
        self.integrator = integrator
        self.net = net

        self.gamma = 0.1 #rate of learning rate decrease.

        self.clip = 0.001 #gradient clipping

        self.optimiser = Adam(self.net.parameters(), lr=self.lr)
        self.milestone = 400 #how often to decay learning rate

        #COSINE ANNEALING LR SCHEDULE https://medium.com/@utkrisht14/cosine-learning-rate-schedulers-in-pytorch-486d8717d541
        self.scheduler1 = CosineAnnealingLR(self.optimiser, 
                                        T_max=100, eta_min=1e-10)
        #self.scheduler = ExponentialLR(self.optimiser, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".
        self.scheduler2 = ReduceLROnPlateau(self.optimiser,
                                            mode="min",
                                            factor=0.1,
                                            patience=10, #how many steps to wait before 
                                            eps=1e-8,
                                            cooldown=50) #Decays the learning rate based on a passed metric
        
    def train_step(self):  
        
        self.net.train() #set neural net into train mode

        #reset gradients 
        self.optimiser.zero_grad()
        
        Lhat = self.integrator.loss_fn()
        clip_grad_norm_(self.net.parameters(), self.clip)

        ##training##
        Lhat.backward()
        self.optimiser.step()
            
        return Lhat.detach().numpy().item()

class TrainRho():
    """
    Defines the training loop for the loss function 
    for learning boundary conditions
    
    overwrite any training params here
    
    args
        rhonet: neural network used to approximate the final distribtuion
        integrator: class that holds integration functions for the fokker planck
    """

    def __init__(self, rhonet):#, integrator):

        self.rhonet = rhonet 

        self.lr = 0.01 #initial learning rate
        #self.integrator = integrator

        self.gamma = 0.1 #rate of learning rate decrease.

        self.clip = 0.01 #gradient clipping

        self.rho_epochs = 2048 #total number of training epochs
        self.milestone = 512 #how often to decay learning rate
        self.rho_batch = 128 #size of batch 

        self.loss = torch.nn.MSELoss()

    def generate_training_data(self,x0,integrator):
        
        with torch.no_grad():

            yobs = integrator.compute_pdf(x0)

        return yobs

    def train_step(self,optimiser,integrator):  

        #reset gradients 
        optimiser.zero_grad()
        
        #initialise x values 
        x0 = torch.tensor(integrator.boundary.generate_samples(self.rho_batch,mode="initial"),dtype=torch.float32,requires_grad=True) 
        # (-5 - 5)*torch.rand(self.rho_batch, 2, dtype=torch.float32,requires_grad=True) + 5

        yobs = self.generate_training_data(x0.clone().detach(),integrator)

        ypred = self.rhonet(x0)
        Lhat = self.loss(ypred,yobs)
        clip_grad_norm_(self.rhonet.parameters(), self.clip)
             
        # Do backprop and optimizer step
        if ~(torch.isnan(Lhat) | torch.isinf(Lhat)):
            ##training##
            Lhat.backward()
            optimiser.step()
            
        else:
            raise ValueError('Girsanov net: Loss function is NaN. Terminating training process')

        return #loss_out


    def train_loop(self, integrator, rho_epochs=None):
        """
        loop for training the rho neural net
        args:
            epochs:
                number of epochs to train the network between updates to the drift net
        """

        if rho_epochs==None:
            rho_epochs=self.rho_epochs
        
        #reset optimiser and scheduler for each train loop
        optimiser = Adam(self.rhonet.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimiser, 
                            T_max=100, eta_min=1e-10)
                            #ExponentialLR(optimiser, gamma=self.gamma)
        self.rhonet.train() #set neural net into train mode
        
        with tqdm(total=int(round(rho_epochs/self.milestone)), bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
                postfix=["Batch", {"value": 0}]) as t:
            for rho_epoch in range(int(round(rho_epochs/self.milestone))):    
            
                for __ in range(self.milestone):
                    self.train_step(optimiser,integrator) #repeat training step
            
                    scheduler.step() #decrease learning rate
                
                t.postfix[1]["value"] = rho_epoch
                t.update()

        self.rhonet.eval() #return neural net into eval mode

if __name__=="__main__":
    Pretrain(path="src/refsolution/UnderdampedHalfNet/refsol.csv").get_pretraining_data()