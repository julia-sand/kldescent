import model
import numpy as np
import torch
from torch import nn
import torch.optim as optim

### train neural net to approximate the numerical 
# perturbative drift from CSV data  

#define defaults
#  
#TODO: train the neural network with the real perturbative data (works only for dummy data atm.)
def pretraining_perturb(net,path,n_epochs=1):   
    
    #n_epochs = 200
    loss_fn = nn.MSELoss()  
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    #is it worth scheduling the learning rate? https://jamesmccaffrey.wordpress.com/2020/12/08/pytorch-learning-rate-scheduler-example/

    ##fetch the perturbative underdamped approximation data
    arr = np.loadtxt(path,
                    delimiter=",", skiprows=1)
    
    #arr = arr[np.where(np.abs(arr[:,1])<3.7)] #drop the edges
    X = torch.tensor(arr[:,0:2], dtype=torch.float32)
    y = torch.tensor(arr[:,-1], dtype=torch.float32).reshape(-1, 1)

    for epoch in range(n_epochs):
        y_pred = net(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    #make trained checkpoint 
    #torch.save(net.state_dict(),"net_pretrained.pt")

if __name__ == "__main__":
  import model
  net = model.NeuralNetwork().to("cpu")
  net.load_state_dict(torch.load("kldescent/tests/net_pretrained.pt", weights_only=True))
  pretraining_perturb(net,2)
  import viz.plots.PlotFuns as plots
  plots.plot_drift(net)

