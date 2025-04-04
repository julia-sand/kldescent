import torch
from torch import nn


#https://www.geeksforgeeks.org/swish-activation-function-in-pytorch/
# Swish Function 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# FROM TORCH TUTORIAL
#create a simple neural network to parametrise the drift. 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 10),  #two inputs
            Swish(),
            nn.Linear(10, 4),   #hidden layer
            Swish(),
            nn.Linear(4, 1)     #one output
        )        

    def forward(self, x):   #how the model should evaluate
        x = self.flatten(x)
        out = self.linear_stack(x)
        return out
