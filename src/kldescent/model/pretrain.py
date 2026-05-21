import numpy as np
import torch
import matplotlib.pyplot as plt

#from torch import nn as torchnn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
        

class Pretrain():
    """
    Class for training a neural network on data stored in CSV format.
        
    Used for eg. initialising a drift or training a secondary model to interpolate 
    benchmark perturbative evolutions
    """
    
    def __init__(self,folder_name,epsilon=None,path_to_pretrain=None):
        r"""
        Define the training parameters and loop for pretraining.

        Parameters
        ----------
        folder_name : str
            Path to the folder containing the CSV training data. 
            By default, 'perturbative_underdamped.csv' is used  

        epsilon : float, optional
            Small parameter used for transformation.
            If None, no transformation is applied.

        path_to_pretrain : str, optional
            Overwrite the default behaviour by providing a filepath for a csv 
            to be used in training.
        """

        self.epsilon = epsilon 
        self.folder_name = folder_name
        
        if path_to_pretrain is None:    
            self.path = folder_name+"/perturbative_underdamped.csv"
        else:
            self.path = path_to_pretrain#folder_name+"/"+path_to_pretrain
        
        #set defaults of training process
        self.lr = 0.001 #initial learning rate
        self.gamma = 0.9 #decay rate of learning rate
        self.milestone = 1000 #frequency of learning rate decay (in epochs)
        self.n_epochs = 3000 #total number of epochs
        self.minibatch_samples = 128 #number of datarows to use per parameter update
        self.clip = 0.5 #size of gradient clipping

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
        
        y = torch.tensor(arr[:,-1], dtype=torch.float32).reshape(-1, 1)
        
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
        optimiser = Adam(net.FeedForwardLayers.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimiser, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".

        X,y = self.get_pretraining_data()

        net.train()
        
        loss_history = []
        
        for epoch in tqdm(range(1,self.n_epochs+1)):

            #generate random batch 
            idx = torch.randint(low=0,high=X.shape[0],size=(self.minibatch_samples,))

            y_pred = net(X[idx,:])

            loss = loss_fn(y_pred,y[idx])
            
            if torch.isfinite(loss):
                optimiser.zero_grad()

                loss.backward()
                clip_grad_norm_(net.parameters(), self.clip)

                optimiser.step()       
            
            optimiser.zero_grad()
            
            #print(y_pred.mean().item(), y_pred.std().item())
            #print(y[idx].mean().item(), y[idx].std().item())
            #print(f'Finished epoch {epoch}, loss {loss}')
            
            loss_history.append(loss.item())
            
            if epoch % self.milestone == 0:
                scheduler.step()
                with torch.no_grad():    
                    net.eval()
                    print(f'Finished epoch {epoch}, loss {loss}')
                    self.plot_training_process_drift(X.clone(),y.clone(),net,epoch,0)
                    #self.plot_training_process_drift(X.clone(),y.clone(),net,epoch,5)
                    net.train()

    @torch.no_grad
    def plot_training_process_drift(self,X,y,net,epoch,timet0):
        net.eval()

        plt.plot(X[X[:,0]==timet0,1].cpu().numpy().flatten(),y[X[:,0]==timet0].cpu().numpy().flatten())
        plt.plot(X[X[:,0]==timet0,1].cpu().numpy().flatten(),net(X[X[:,0]==timet0,:]).cpu().numpy().flatten())

        plt.savefig(self.folder_name+f"/training_{epoch}_.png")
        plt.close("all")

    @torch.no_grad
    def plot_pretrain_loss(self,loss_history):
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.savefig(self.folder_name+f"/loss_pretrain.png")
        plt.close()

def run_pretrain(net,
                    folder_name,
                    path_to_pretrain,
                    epsilon=None):
    r"""
    Trains a neural network to approximate a numerical function given by values in a csv

    args:
        net :
            neural network to be trained
        folder_name : str
            folder location to save intermediate training plots
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
    
    pretrainer = Pretrain(folder_name=folder_name,epsilon=epsilon,path_to_pretrain=path_to_pretrain)
    pretrainer.pretraining(net)
    
    print("Pretraining completed successfully.")

if __name__=="__main__":
    run_pretrain()
