import os

import numpy as np 
import torch
from torch import nn as torchnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR,ExponentialLR
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm#import pdb

class Pretrain():
    #define the training parameters and loop for any pretraining. 
    def __init__(self,path):
        self.path = path 

        #set defaults of training process
        self.lr = 0.001 #initial learning rate
        self.gamma = 0.1 #decay rate of learning rate
        self.step_size = 500 #frequency of learning rate decay (in epochs)
        self.n_epochs = 1000 #total number of epochs

        self.clip = 0.1

    def get_pretraining_data(self):
        #load csv and return data 

        arr = np.loadtxt(self.path,
                        delimiter=",", skiprows=1)
        
        #pdb.set_trace()
        X = torch.tensor(arr[:,0:2], dtype=torch.float32)
        y = torch.tensor(arr[:,-1], dtype=torch.float32).reshape(-1, 1)
        
        return X,y/(2**0.5)

    def pretraining(self,net):   
        
        #set up training parameters
        loss_fn = torchnn.MSELoss()  
        optimizer = Adam(net.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".

        X,y = self.get_pretraining_data()

        for epoch in range(1,self.n_epochs+1):
            
            #pdb.set_trace()
            optimizer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y)
            #clip_grad_norm_(net.parameters(), self.clip)
            loss.backward()
            optimizer.step()       

            if epoch % self.step_size == 0:
                scheduler.step()
                print(f'Finished epoch {epoch}, loss {loss}')
        
class Train():
    """
    Defines the parameters of the training process and 
    the training step
    """

    def __init__(self, net, integrator):

        #set defaults of training process
        self.lr = 0.005 #initial learning rate
        self.integrator = integrator
        self.net = net

        self.gamma = 0.1 #rate of learning rate decrease.

        self.clip = 0.001 #gradient clipping

        self.optimiser = Adam(self.net.parameters(), lr=self.lr)
        self.milestone = 100 #how often to decay learning rate

        #COSINE ANNEALING LR SCHEDULE https://medium.com/@utkrisht14/cosine-learning-rate-schedulers-in-pytorch-486d8717d541
        self.scheduler = CosineAnnealingLR(self.optimiser, 
                            T_max=100, eta_min=1e-10)
        #self.scheduler = ExponentialLR(self.optimiser, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".

        
    def train_step(self):  
        
        self.net.train() #set neural net into train mode

        #reset gradients 
        self.optimiser.zero_grad()
        
        Lhat = self.integrator.loss_fn()
        clip_grad_norm_(self.net.parameters(), self.clip)

        ##training##
        Lhat.backward()
        self.optimiser.step()
               
        self.scheduler.step()
        #if (epoch % self.milestone == 0) and (epoch>0):
            
        return Lhat.detach().numpy().item()


class Metrics():
    """
    Class for finding metrics during the training process 

    Method: 
        update_metrics
    """

    def __init__(self, net, integrator, logstep, logger):
        self.integrator = integrator
        self.net = net
        #self.rhonet = rhonet
        #self.scheduler = trainer.scheduler # for testing out different schedules, we record the learning rate 
        
        self.logstep = logstep #how frequently to save results
        self.logger = logger
        self.update_metrics(0, 0)

    def update_metrics(self, epoch, loss_out, trainer=None):
        """
        Function for computing metrics
            -logs the metrics at logstep
            - checks for nan loss
            TODO (fix logic)- updates the metrics for final distribution, gaussian reference. 
        args:
            epoch:
                pass current epoch to use in logging loss
            loss_out:
                value of the loss from the metrics function
        """
        
        with torch.no_grad():

            if loss_out is not None:
                loss_out = loss_out
            else:
                loss_out = 0 #TODO: compute the loss independently here

            if np.isnan(loss_out):
                raise ValueError('Loss function is NaN. Terminating training process')
        
            if (epoch % self.logstep == 0): #compute metrics 
                with torch.no_grad():
                    #this also updates the ref. cumulants
                    kl_dynamic, kl_static, mean0, var0 =self.integrator.compute_metrics()

                    if trainer is not None:
                        lr_rate = trainer.scheduler.get_last_lr()[0]
                    else:
                        lr_rate = 0

                    self.logger.write_data_row(epoch,
                                                loss_out,
                                                kl_dynamic,
                                                kl_static,
                                                lr_rate, #,
                                                mean0,
                                                var0)

                    self.logger.save_checkpoint(self.net,epoch)
                    #if self.rhonet is not None:
                    #    self.logger.save_checkpoint(self.rhonet,epoch)
            else:
                pass
            #else: 
            #    with torch.no_grad():
            #        self.integrator.update_ref_statistics()
                    #pass

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

        self.lr = 0.001 #initial learning rate
        #self.integrator = integrator

        self.gamma = 0.1 #rate of learning rate decrease.

        self.clip = 0.01 #gradient clipping

        self.rho_epochs = 1000 #total number of training epochs
        self.milestone = 200 #how often to decay learning rate
        #self.fpk_integrator = FokkerPlanck(integrator.boundary,integrator.net)
        
        self.loss = torch.nn.MSELoss()

    def generate_training_data(self,x0,integrator):
        
        with torch.no_grad():

            yobs = integrator.compute_pdf(x0)

        return yobs


    def train_step(self,optimiser,integrator):  

        #reset gradients 
        optimiser.zero_grad()
        
        #initialise x values 
        x0 = (-4 - 4)*torch.rand(500, 2, dtype=torch.float32,requires_grad=True) + 4

        yobs = self.generate_training_data(x0,integrator)

        ypred = self.rhonet(x0)
        Lhat = self.loss(ypred,yobs)
        #clip_grad_norm_(self.rhonet.parameters(), self.clip)

        ##training##
        Lhat.backward()
        optimiser.step()
        
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
        scheduler = ExponentialLR(optimiser, gamma=self.gamma)
        self.rhonet.train() #set neural net into train mode
        
        with tqdm(total=int(round(rho_epochs/self.milestone)), bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
                postfix=["Batch", {"value": 0}]) as t:
            for rho_epoch in range(int(round(rho_epochs/self.milestone))):    
            
                for __ in range(self.milestone):
                    self.train_step(optimiser,integrator) #repeat training step
            
                scheduler.step() #decrease learning rate
                
                t.postfix[1]["value"] = rho_epoch
                t.update()

if __name__=="__main__":
    Pretrain("src/refsolution/UnderdampedHalfNet/refsol.csv").get_pretraining_data()