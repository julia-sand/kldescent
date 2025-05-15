import torch
from torch import nn
import os
#from einops import rearrange

#https://www.geeksforgeeks.org/swish-activation-function-in-pytorch/
# Swish Function 
class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

#taken from qsh-zh/pis
#time encoding 
class TimeConder(nn.Module):
    def __init__(self, 
                    channel, #input size
                    out_dim=64, #output size 
                    num_layers=2 #number of layers in the network
                    ): 
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        #cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(sin_cond,cos_cond)

class MyActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# FROM TORCH TUTORIAL
#create a simple neural network to parametrise the drift. 
class NeuralNetwork(nn.Module):
    def __init__(self,hiddenlayerneurons,
                        boundary,
                        checkpointpath=None):

        #params
        self.hidden_layer_neurons = hiddenlayerneurons 
        self.boundary = boundary

        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        #self.tencoding = TimeConder(1,self.hidden_layer_neurons) #fourier positional encoding goes here! 
        self.NNt = nn.Sequential(
            nn.Linear(1, self.hidden_layer_neurons),  #
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #64 outputs
            nn.ReLU(), #ensure these are positive outputs 
            ) 
        self.NNx = nn.Sequential(
            nn.Linear(1, self.hidden_layer_neurons),  #input x
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #64 outputs
            )    
        self.NN1 = nn.Sequential(
            nn.Linear(2*self.hidden_layer_neurons, self.hidden_layer_neurons),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #64 outputs
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, 1))   #64 outputs)  
        self.NN2 = nn.Sequential(
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #64 outputs
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, 1),   #64 outputs
            nn.ReLU(), #ensure these are positive outputs 
            )  

    def forward(self, input_vec):   #how the model should evaluate
        
        x = self.NNx(input_vec[:,1,None]) #do the time encoding on the parameter t
        t = self.NNt(input_vec[:,0,None])
        #self.NNt(self.tencoding(input_vec[:,0].reshape((input_vec.shape[0],1))))

        #concatenate the outputs and put them through the part 2
        out = self.NN1(torch.column_stack((t,x))) + self.NN2(t)*self.boundary.scorefinal(input_vec[:,1,None])
        
        return out

    def initialize_weights(self,net):   #sets all the weights away from zero.
        for m in net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,gain=4)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)



# FROM TORCH TUTORIAL
#create a simple neural network to parametrise the drift. 
class MLP(nn.Module):
    def __init__(self,hiddenlayerneurons,checkpointpath=None):
        self.hidden_layer_neurons = hiddenlayerneurons 

        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, self.hidden_layer_neurons),  #two inputs
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            MyActivation(),
            nn.Linear(self.hidden_layer_neurons, 1)     #one output
        )    
        #load checkpoint
        #if checkpointpath is not None:
        #    if checkpointpath=="zeros":
        #        self.initialize_weights()
        #        print("Initial weights set to zero.")  
        #    else:
        #        self.load_state_dict(torch.load(os.path.dirname(__file__)+"/../.."+checkpointpath, weights_only=True))
        #        print("Checkpoint loaded successfully.")    
        #else:
        #    print("Neural Network initialised using default method.")  

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
