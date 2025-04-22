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
    
    loss_fn = nn.MSELoss()  
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
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

def preprocess():
    ##this is a script to preprocess the results of 
    from scipy.ndimage import generic_filter,median
    import os 
    dirname = os.path.dirname(__file__) #fetches current location 
    refsolfile=dirname+"/refsolution/value_orig.csv"
    
    ##fetch the perturbative underdamped approximation data
    arr = np.loadtxt(refsolfile,
                        delimiter=",")  

    times = np.unique(arr[:,0])
    for t in range(len(times)):  
        filter_delta = 200
        valtemp = arr[np.where(arr[:,0]==times[t]),-2].reshape(5000) 
        qtemp = arr[np.where(arr[:,0]==times[t]),1].reshape(5000) 
        val_out = generic_filter(valtemp,median,filter_delta,mode="nearest")
        arr[np.where(arr[:,0]==times[t]),-2] = val_out
        dvout = generic_filter(np.gradient(val_out[filter_delta:-filter_delta],qtemp[filter_delta:-filter_delta]),median,filter_delta,mode="nearest") 
        arr[np.where(arr[:,0]==times[t]),-1] = np.interp(qtemp,qtemp[filter_delta:-filter_delta],dvout,left=dvout[0],right=dvout[-1])
        #generic_filter(,median,500,mode="nearest")  

    np.savetxt(dirname+"/refsolution/value.csv",arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')

    return 



if __name__ == "__main__":
  
  preprocess()
  