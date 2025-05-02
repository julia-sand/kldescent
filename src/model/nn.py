import torch
from torch import nn
import os

#https://www.geeksforgeeks.org/swish-activation-function-in-pytorch/
# Swish Function 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# FROM TORCH TUTORIAL
#create a simple neural network to parametrise the drift. 
class NeuralNetwork(nn.Module):
    def __init__(self,hiddenlayerneurons,checkpointpath=None):
        self.hidden_layer_neurons = hiddenlayerneurons 

        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, self.hidden_layer_neurons),  #two inputs
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, self.hidden_layer_neurons),   #hidden layer
            nn.ReLU(),
            nn.Linear(self.hidden_layer_neurons, 1)     #one output
        )    
        #load checkpoint
        if checkpointpath is not None:
            if checkpointpath=="zeros":
                self.initialize_weights()
                print("Initial weights set to zero.")  
            else:
                self.load_state_dict(torch.load(os.path.dirname(__file__)+"/../.."+checkpointpath, weights_only=True))
                print("Checkpoint loaded successfully.")    
        else:
            print("Neural Network initialised using default method.")  
            
          

    def forward(self, x):   #how the model should evaluate
        x = self.flatten(x)
        out = self.linear_stack(x)
        return out

    def initialize_weights(self):   #sets all the weights to zero.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
