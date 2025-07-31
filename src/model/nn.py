import os
import math

import torch
from torch import nn

from ..utils.parser import network_params

#import pdb

#https://www.geeksforgeeks.org/swish-activation-function-in-pytorch/
# Swish Function 
class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

class MyActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

##FROM: https://github.com/miaotianyi 
class FourierFeatures(nn.Module):
    def __init__(self, out_features: int,
                 sigma: float = 1.,
                 trainable: bool = False,
                 device=None, dtype=None):
        """
        For input feature vector `x`, apply linear transformation
        followed by concatenated sine and cosine: `[sin(Ax), cos(Ax)]`.

        - Input: `(*, in_features)`, where the last dimension is the feature dimension
        - Output: `(*, out_features)`, where all but the last dimension are the same shape as the input

        Unlike element-wise sinusoidal encoding,
        Fourier features capture the interaction of high-dimensional input features.

        Note that a bias term is not necessary in the linear transformation,
        because `sin(wx+b)` and `cos(wx+b)` can both be represented by
        linear combinations of `sin(wx)` and `cos(wx)`, which can be learned in the
        subsequent linear layer.
        (https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

        In learnable Fourier features (https://arxiv.org/abs/2106.02795),
        the linear transformation is trainable after random normal initialization.
        They also use an MLP (linear, GELU, linear) following this Fourier feature layer,
        because the goal is to produce a positional encoding added to word embedding.
        We do not implement the MLP here, since a user should be free to decide
        which layers to follow; this keeps the modularity of FourierFeatures.

        Also note that in learnable Fourier features, the final output
        after sine/cosine is divided by sqrt(H_out). We do not implement that yet.

        Parameters
        ----------
        in_features: int
            Size of each input sample

        out_features: int
            Size of each output sample

        sigma: float
            Standard deviation of 0-centered normal distribution,
            which is used to initialize linear transformation parameters.

        trainable: bool, default: False
            If True, the linear transformation is trainable by gradient descent.
        """
        super().__init__()
        assert out_features % 2 == 0, "number of out_features must be even"
        self.in_features = 1 #in_features
        self.out_features = out_features
        self.sigma = sigma  # standard deviation for normal initialization
        self.trainable = bool(trainable)

        # define linear layer
        self.linear = nn.Linear(
            in_features=self.in_features, out_features=out_features // 2,
            bias=False,
            device=device, dtype=dtype)

        # initialize weights
        nn.init.normal_(self.linear.weight.data, mean=0.0, std=self.sigma)

        # freeze layer if not trainable
        for param in self.linear.parameters():
            param.requires_grad_(self.trainable)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        #pdb.set_trace()
        t = self.linear(t)
        t = 2 * math.pi * t
        t = torch.cat((t.sin(), t.cos()), dim=-1)
        return t

class NeuralNetwork(nn.Module):
    def __init__(self,boundary,nn_params=None,logger=None):
        
        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = network_params()
        
        if logger is not None:
            logger.log_dict(self.params)

        #params
        self.boundary = boundary

        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        #self.tencoding = TimeConder(1,self.hidden_layer_neurons) #fourier positional encoding goes here! 
        self.NNt = nn.Sequential(
            nn.Linear(2*self.params["hiddenlayerneurons"], self.params["hiddenlayerneurons"] ),  #
            nn.GELU(), #MyActivation(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            nn.ReLU(),  #must be positive
            ) 
        self.NNx = nn.Sequential(
            nn.Linear(1, self.params["hiddenlayerneurons"] ),  #input x
            nn.GELU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            )    
        self.NN1 = nn.Sequential(
            nn.Linear(2*self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            nn.GELU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            nn.GELU(),
            nn.Linear(self.params["hiddenlayerneurons"] , 1))  
        self.NN2 = nn.Sequential(
            nn.Linear(2*self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            nn.GELU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            nn.GELU(),
            nn.Linear(self.params["hiddenlayerneurons"] , 1),   #64 outputs
            nn.ReLU(), #must be positive
            )  
        self.FourierFeatures = FourierFeatures(2*self.params["hiddenlayerneurons"])
        
        if self.params["checkpointpath"] is not None: #automatically loads a checkpoint from the command line
            self.load_checkpoint(self.params.checkpointpath)

    def forward(self, input_vec):   #how the model should evaluate
        #pdb.set_trace()

        x = self.NNx(input_vec[...,[1]])
        t = self.FourierFeatures(input_vec[...,[0]])#
        #self.NNt(input_vec[...,[0]])
        tout = self.NNt(t)
        #concatenate the outputs and put them through the part 2
        out = self.NN1(torch.column_stack((tout,x))) + self.NN2(t)*self.boundary.scorefinal(input_vec[...,[1]])
        
        return out

    def initialize_weights(self,net):   #sets all the weights away from zero.
        for m in net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,gain=2)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def load_checkpoint(self,checkpointpath):
        
        try:
            checkpointpath = os.getcwd() + "/" + checkpointpath
            self.load_state_dict(torch.load(checkpointpath, weights_only=True))
            print("Checkpoint loaded successfully.")    
        except FileNotFoundError:
            print("The checkpoint file was not found.  \n \
                    An epoch with a saved checkpoint must be specified. Hint: The filepath must be provided from the current working directory. Use --epoch to specify which epoch.")
            raise FileNotFoundError
        except AttributeError:
            print("Checkpoint could not be loaded. Checkpoint format not compatible with initialised network. Continuing with default initialisation-")
            #raise AttributeError
            pass

class RhoNeuralNetwork(nn.Module):
    """
    Neural Network object for parametrising the form of the probability distribution
    """

    def __init__(self,boundary,nn_params=None,logger=None):
        
        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = network_params()
        
        if logger is not None:
            logger.log_dict(self.params)

        #params
        self.boundary = boundary

        super(RhoNeuralNetwork, self).__init__()
        self.NNx = nn.Sequential(
            nn.Linear(2, self.params["hiddenlayerneurons"] ),  #input x
            nn.GELU(),
            nn.Linear(self.params["hiddenlayerneurons"] , self.params["hiddenlayerneurons"] ),   #64 outputs
            nn.ReLU(), #enforce positive output since we want to learn a pdf
            )  

    def forward(self, x0):   #how the model should evaluate
        
        return self.NNx(x0)


    def initialize_weights(self,net):   #sets all the weights away from zero.
        for m in net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,gain=2)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def load_checkpoint(self,checkpointpath):
        #TODO update this function to look for a separate rho nn checkpoint.
        try:
            checkpointpath = os.getcwd() + "/" + checkpointpath
            self.load_state_dict(torch.load(checkpointpath, weights_only=True))
            print("Checkpoint loaded successfully.")    
        except FileNotFoundError:
            print("The checkpoint file was not found.  \n \
                    An epoch with a saved checkpoint must be specified. Hint: The filepath must be provided from the current working directory. Use --epoch to specify which epoch.")
            raise FileNotFoundError


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



if __name__ == "__main__":
    from torchinfo import summary

    from ..sde.boundary import UnderdampedHalfGaussian
    boundary_cond = UnderdampedHalfGaussian() #getattr(boundaries,run_params["boundary"])()
    
    net = NeuralNetwork(boundary_cond)

    summary(net, input_size=(1,2))