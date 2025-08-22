import os

import numpy as np 
import torch
from torch import nn as torchnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR,ExponentialLR
from torch.nn.utils import clip_grad_norm_

#torch.autograd.set_detect_anomaly(True)
#from memory_profiler import profile

class Pretrain():
    #define the training parameters and loop for any pretraining. 
    def __init__(self,path):
        self.path = path 

        #set defaults of training process
        self.lr = 0.01 #initial learning rate
        self.gamma = 0.1 #decay rate of learning rate
        self.step_size = 10000 #frequency of learning rate decay (in epochs)
        self.n_epochs = 40000 #total number of epochs
      
    def save_epoch_trained_data(self,epoch):
        
        return

    def get_pretraining_data(self):
        #load csv and return data 

        arr = np.loadtxt(self.path,
                        delimiter=",", skiprows=1)
        
        X = torch.tensor(arr[:,0:2], dtype=torch.float32)
        y = torch.tensor(arr[:,-1], dtype=torch.float32).reshape(-1, 1)
        return X,y 

    def pretraining_perturb(self,net):   
        
        #set up training parameters
        loss_fn = torchnn.MSELoss()  
        optimizer = Adam(net.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".

        X,y = self.get_pretraining_data()

        for epoch in range(1,self.n_epochs+1):
            optimizer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y)
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
        
        Lhat = self.integrator.loss_fn(self.net)
        clip_grad_norm_(self.net.parameters(), self.clip)

        ##training##
        Lhat.backward()
        self.optimiser.step()
               
        self.scheduler.step()
        #if (epoch % self.milestone == 0) and (epoch>0):
            
        loss_out = Lhat.detach().numpy().item()
        return loss_out


class Metrics():
    """
    Class for finding metrics during the training process 

    Method: 
        update_metrics
    """

    def __init__(self, trainer, logstep):
        #use trainer class to initialise 
        self.integrator = trainer.integrator
        self.net = trainer.net
        self.scheduler = trainer.scheduler # for testing out different schedules, we record the learning rate 
        self.logstep = logstep #how frequently to save results
    
    def update_metrics(self, epoch, loss_out, logger):
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
            if np.isnan(loss_out):
                raise ValueError('Loss function is NaN. Terminating training process')
            
            if (epoch % self.logstep == 0): #compute metrics 
                with torch.no_grad():
                    #this also updates the ref. cumulants
                    kl_dynamic, kl_static, mean0, var0 =self.integrator.compute_metrics(self.net)

                    logger.write_data_row(epoch,
                                                loss_out,
                                                kl_dynamic,
                                                kl_static,
                                                self.scheduler.get_last_lr()[0],
                                                mean0,
                                                var0)

                    logger.save_checkpoint(self.net,epoch)
        
            else: 
                with torch.no_grad():
                    self.integrator.update_ref_statistics(self.net)



class TrainRho():
    """
    Defines the training loop for the loss function 
    for learning boundary conditions
    
    overwrite any training params here
    
    args
        rhonet: neural network used to approximate the final distribtuion
        integrator: class that holds integration functions for the fokker planck
    """

    def __init__(self, rhonet, integrator):

        self.rhonet = rhonet 

        self.lr = 0.005 #initial learning rate
        self.integrator = integrator

        self.gamma = 0.1 #rate of learning rate decrease.

        self.clip = 0.001 #gradient clipping

        self.optimiser = Adam(self.rhonet.parameters(), lr=self.lr)
        self.rho_epochs = 100 #total number of training epochs
        self.milestone = 20 #how often to decay learning rate

        #COSINE ANNEALING LR SCHEDULE https://medium.com/@utkrisht14/cosine-learning-rate-schedulers-in-pytorch-486d8717d541
        #self.scheduler = CosineAnnealingLR(self.optimiser, T_max=100, eta_min=1e-10)
        self.scheduler = ExponentialLR(self.optimiser, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".
        
    def train_step(self):  
        
        self.rhonet.train() #set neural net into train mode

        #reset gradients 
        self.optimiser.zero_grad()
        
        Lhat = self.integrator.loss_fn(self.rhonet)
        clip_grad_norm_(self.rhonet.parameters(), self.clip)

        ##training##
        Lhat.backward()
        self.optimiser.step()
               
        #self.scheduler.step()
            
        print(Lhat.detach().numpy().item())
        return #loss_out

    def train_loop(self, epochs=None):
        """
        loop for training the rho neural net
        args:
            epochs:
                number of epochs to train the network between updates to the drift net
        """

        if epochs==None:
            epochs=self.rho_epochs


        for _ in range(int(round(epochs/self.milestone))):    
            for __ in range(self.milestone):
                self.train_step() #repeat training step
            
            self.scheduler.step() #decrease learning rate
        
   
    def update_metrics(self, epoch, loss_out, logger):
        """
        Function for computing metrics and saving checkpoints of the rhonet
        args:
            epoch:
                pass current epoch to use in logging loss
            loss_out:
                value of the loss from the metrics function
        """
        
        with torch.no_grad():
            if np.isnan(loss_out):
                raise ValueError('Loss function is NaN. Terminating training process')
            
            if (epoch % self.logstep == 0): #compute metrics 
                with torch.no_grad():
                    self.logger.save_checkpoint(self.rhonet,epoch)
        
            else: 
                with torch.no_grad():
                    self.integrator.update_ref_statistics(self.net)

