import os

import numpy as np 
import torch
#from torch import nn as torchnn
from torch.optim import Adam,SGD,AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR,ExponentialLR,ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

class Pretrain():
    """
    Class for training a neural network to learn CSV data
        
    Used for eg. initialising a drift or training a secondary model to interpolate 
    benchmark perturbative evolutions
    """
    
    def __init__(self,epsilon,path_to_pretrain):
        r"""
        Define the training parameters and loop for any pretraining. 

        args:
            epsilon : float
                if None, no transformation is done
            path_to_pretrain : str
                location of the CSV to train
        """
        
        self.epsilon = epsilon 
        self.path = path_to_pretrain 

        #set defaults of training process
        self.lr = 0.0001 #initial learning rate
        self.gamma = 0.9 #decay rate of learning rate
        self.milestone = 1000 #frequency of learning rate decay (in epochs)
        self.n_epochs = 10000 #total number of epochs

        self.minibatch_samples = 128 #number of datarows to use per parameter update

        self.clip = 0.1 #size of gradient clipping

    def get_pretraining_data(self):
        """Load data from csv to be used in training the network
        Data is transformed when a value for epsilon is passed into the class.
        If no transformation is needed, set epsilon=None at class level. 
        """ 

        #load csv 
        arr = np.loadtxt(self.path,
                        delimiter=",", skiprows=1)
        
        #remove zero probability areas
        #arr = arrtemp[np.where(arrtemp[:,2]>0)] 

        X = torch.tensor(arr[:,0:2], dtype=torch.float32)
        
        y = torch.tensor(arr[:,-2], dtype=torch.float32).reshape(-1, 1)
        
        if self.epsilon is not None:

            X,y = self.transform_data(X,y)

        return X,y

    def transform_data(self,X,y):
        r"""Implement in place transformation for the data, which corresponds to
        
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

        #X[:,0] /= self.epsilon**2 #get t0 from t2
        y *= (-(self.epsilon/(2**0.5))) #transformation (**)
        
        return X,y 

    def pretraining(self,net):   
        
        #set up training parameters
        loss_fn = torch.nn.MSELoss()  
        optimiser = Adam(net.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimiser, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".

        X,y = self.get_pretraining_data()

        net.train()

        for epoch in tqdm(range(1,self.n_epochs+1)):
                
            #generate random batch 
            idx = torch.randint(low=0,high=X.shape[0],size=(self.minibatch_samples,))

            y_pred = net(X[idx,:])
            loss = loss_fn(y_pred,y[idx])
            clip_grad_norm_(net.parameters(), self.clip)
            
            loss.backward()
            optimiser.step()       
            optimiser.zero_grad()
            #print(y_pred.mean().item(), y_pred.std().item())
            #print(y[idx].mean().item(), y[idx].std().item())
            #print(f'Finished epoch {epoch}, loss {loss}')
            
            if epoch % self.milestone == 0:
                scheduler.step()
                with torch.no_grad():    
                    net.eval()
                    print(f'Finished epoch {epoch}, loss {loss}')
                    self.plot_training_process_drift(X.clone(),y.clone(),net,epoch)
                net.train()

    @torch.no_grad
    def plot_training_process_drift(self,X,y,net,epoch):
        import matplotlib.pyplot as plt
        net.eval()
        #plt.scatter(net(X[X[:,0]==5,:]).cpu().numpy().flatten(),y[X[:,0]==5].cpu().numpy().flatten())
        plt.plot(X[X[:,0]==5,1].cpu().numpy().flatten(),y[X[:,0]==5].cpu().numpy().flatten())
        plt.plot(X[X[:,0]==5,1].cpu().numpy().flatten(),net(X[X[:,0]==5,:]).cpu().numpy().flatten())
        #plt.ylim((-5,5))
        plt.savefig(f"training_{epoch}_.png")
        plt.close("all")

def run_pretrain(net,
                    path_to_pretrain,
                    epsilon=None):
    r"""
    Trains a neural network to approximate a numerical function given by values in a csv

    args:
        net :
            neural network to be trained
        path_to_pretrain : csv
            path from current working directory to a CSV containing data to be used in the pretraining. The data should have a csv format 
            where the first two entries are the input and the last column should contian the trained values. See PerturbativeBenchmarkSolution folder for some examples.  
        epsilon : float
            If passed, the training data is scaled, see data_transform function in the training class for method.
            Default None (no data transformation). 
    returns:
        none
    """

    print("Starting pre-training")
    
    pretrainer = Pretrain(epsilon=epsilon,path_to_pretrain=path_to_pretrain)
    pretrainer.pretraining(net)
    
    print("Pretraining completed successfully.")

if __name__=="__main__":
    run_pretrain()