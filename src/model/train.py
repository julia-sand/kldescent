import numpy as np 
import torch
from torch import nn as torchnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
class Pretrain():

    def __init__(self,path):
        self.path = path 

        #set defaults of training process
        self.lr = 0.01 #initial learning rate
        self.gamma = 0.1 #decay rate of learning rate
        self.step_size = 2000 #frequency of learning rate decay (in epochs)
        self.n_epochs = 10000 #total number of epochs

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
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".

        X,y = self.get_pretraining_data()

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            if epoch % self.step_size == 0:
                scheduler.step()
            print(f'Finished epoch {epoch}, latest loss {loss}')


class Train():

    def __init__(self,path):
        self.path = path 

        #set defaults of training process
        self.lr = 0.01 #initial learning rate
        self.gamma = 0.1 #decay rate of learning rate
        self.step_size = 2000 #frequency of learning rate decay (in epochs)
        self.n_epochs = 10000 #total number of epochs

    def training_step(self,net):   
        return 
        
