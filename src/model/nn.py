import torch
from torch import nn
import os

from ..utils.parser import network_params

#https://www.geeksforgeeks.org/swish-activation-function-in-pytorch/
# Swish Function 
class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

class MyActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# FROM TORCH TUTORIAL
#create a simple neural network to parametrise the drift. 
class NeuralNetwork(nn.Module):
    def __init__(self,boundary,nn_params=None):
        
        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = network_params()

        #params
        self.boundary = boundary

        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        #self.tencoding = TimeConder(1,self.hidden_layer_neurons) #fourier positional encoding goes here! 
        self.NNt = nn.Sequential(
            nn.Linear(1, self.params["hiddenlayerneurons"] ),  #
            nn.SiLU(), #MyActivation(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            nn.ReLU(),  #must be positive
            ) 
        self.NNx = nn.Sequential(
            nn.Linear(1, self.params["hiddenlayerneurons"] ),  #input x
            nn.SiLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            )    
        self.NN1 = nn.Sequential(
            nn.Linear(2*self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            nn.SiLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            nn.SiLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , 1))   #64 outputs)  
        self.NN2 = nn.Sequential(
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            nn.SiLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            nn.SiLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , 1),   #64 outputs
            nn.ReLU(), #must be positive
            )  

        try:
            if self.params["checkpointpath"] is not None: #automatically loads a checkpoint from the command line
                self.load_checkpoint(self.params.checkpointpath)
        except AttributeError:
            pass
        
        #self.initialize_weights(self,net)
    def forward(self, input_vec):   #how the model should evaluate
        #pdb.set_trace()

        x = self.NNx(input_vec[...,[1]])
        t = self.NNt(input_vec[...,[0]])
        
        #concatenate the outputs and put them through the part 2
        out = self.NN1(torch.column_stack((t,x))) + self.NN2(t)*self.boundary.scorefinal(input_vec[...,[1]])
        
        return out

    def initialize_weights(self,net):   #sets all the weights away from zero.
        for m in net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,gain=4)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def load_checkpoint(self,checkpointpath):
        
        try:
            print(self.params["hiddenlayerneurons"] )
            checkpointpath = os.getcwd() + "/" + checkpointpath
            self.load_state_dict(torch.load(checkpointpath, weights_only=True))
            print("Checkpoint loaded successfully.")    
        except FileNotFoundError:
            print("The checkpoint file was not found. The filepath should be provided from the current working directory.")
            raise FileNotFoundError

# FROM TORCH TUTORIAL
#create a simple neural network to parametrise the drift. 
class MLP(nn.Module):
    def __init__(self,checkpointpath=None):
        self.params = network_params()

        
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, self.params["hiddenlayerneurons"] ),  #two inputs
            MyActivation(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #hidden layer
            MyActivation(),
            nn.Linear(self.params["hiddenlayerneurons"] , 1)     #one output
        )    
        self.initialize_weights()   

    def forward(self, x):   #how the model should evaluate
        x = self.flatten(x)
        out = self.linear_stack(x)
        return out

    def initialize_weights(self):   #sets all the weights to zero.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,gain=4)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)



if __name__=="__main__":
    import pdb


    from ..sde.boundary import UnderdampedHalfGaussian
    from ..utils.parser import integrator_params

    boundaryconditions = UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = NeuralNetwork(boundaryconditions).to("cpu")
    
    #net(torch.column_stack((torch.ones((100,1),requires_grad=False,dtype=torch.float32),
    #    torch.linspace(0,10,steps=100).reshape(100,1))))