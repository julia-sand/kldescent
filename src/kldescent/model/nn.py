import os
import math

import torch
from torch import nn

from kldescent.utils.parser import du_network_params,rho_network_params,fetch_checkpoint_path

def load_checkpoint(net,checkpointpath):
    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(checkpointpath, weights_only=True))
        else:
            net.load_state_dict(torch.load(checkpointpath, weights_only=True,map_location=torch.device('cpu')))
        
        print(net.net_name+f": Checkpoint loaded successfully on {net.device}.")   
    except FileNotFoundError:
        print("The checkpoint file at "+checkpointpath+" was not found")
        print("Hint: The filepath must be provided from the current working directory. Use --epoch to specify which epoch to load when using a default with --folder. For loading a specific checkpoint file with a different filename, use the --checkpointpath option.")
        print("Continuing with default initialisation")
        raise BaseException
    except RuntimeError:
     
        print("The format of the checkpoint file at "+checkpointpath+" did not match the initialised network.")
        print("Continuing with default initialisation")
    
        raise BaseException
        
    except AttributeError:
        print("Checkpoint could not be loaded. Checkpoint format not compatible with initialised network. Continuing with default initialisation-")
        raise BaseException

def save_nn_checkpoint(net,loc):
    """save checkpoint"""

    torch.save(net.state_dict(),loc) 

class FourierFeatures(nn.Module):
    def __init__(self,in_features,out_features, scale, device):
        super().__init__()
       
        assert out_features % 2 == 0, "number of out_features must be even"
        self.device = device

        #initialise randn 
        #self.B = scale*torch.randn(in_features, out_features//2).to(self.device)
        self.B_layer = nn.Linear(in_features, out_features//2, bias=False, dtype=torch.float32).to(self.device)
        for param in self.B_layer.parameters():
            param.requires_grad_(True)
        
        #initialise the weights normally
        nn.init.normal_(self.B_layer.weight.data, mean=0.0, std=scale)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        
        t_proj = 2 * math.pi * self.B_layer(t) #t @ self.B
        return torch.cat((torch.sin(t_proj), torch.cos(t_proj)), dim=-1).to(self.device)


class MyActivation(nn.Module): 
    def forward(self,x):
        return torch.sin(x)

#@torch.compile
class NeuralNetwork_layers(nn.Module):
    def __init__(self,device,hidden_layer_neurons):
        
        self.net_name = "du_layers_only"
               
        super().__init__()

        # store normalization constants (not trainable)
        
        self.flatten = nn.Flatten()
        #self.tencoding = TimeConder(1,self.hidden_layer_neurons) #fourier positional encoding goes here! 
        self.NNt = nn.Sequential(
            nn.Linear(1, hidden_layer_neurons ),  #
            #nn.Linear(2*self.params["duhiddenlayerneurons"], self.params["duhiddenlayerneurons"] ),  #
            nn.GELU(), #MyActivation(),
            nn.Linear(hidden_layer_neurons , hidden_layer_neurons),   #64 outputs
            #nn.GELU(), #MyActivation(),
            #nn.Linear(self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),   #64 outputs
            nn.GELU(),  #must be positive
            ).to(device)
        self.NNx = nn.Sequential(
            nn.Linear(1,hidden_layer_neurons ),  #input x
            nn.GELU(),
            nn.Linear(hidden_layer_neurons, hidden_layer_neurons),   #64 outputs
            ).to(device)   
        self.NN1 = NN2(device,hidden_layer_neurons).to(device) 
        #nn.Sequential(
        #    nn.Linear(2*hidden_layer_neurons, hidden_layer_neurons),  #input 2x64 features: concatenated output of NNt and NNx t and x 
        #    nn.GELU(),
        #    nn.Linear(hidden_layer_neurons,hidden_layer_neurons),   #64 outputs
        #    nn.GELU(),
        #    nn.Linear(hidden_layer_neurons , 1)).to(device) 
        self.NN2 = NN2(device,hidden_layer_neurons).to(device) 
        self.layer_norm_combined = nn.LayerNorm(2*hidden_layer_neurons).to(device)

class NN2(nn.Module):

    def __init__(self,device,hidden_layer_neurons):
        super().__init__()
        self.layer1 = nn.Linear(2*hidden_layer_neurons , hidden_layer_neurons).to(device) 
        self.layer2 = nn.Linear(hidden_layer_neurons , hidden_layer_neurons).to(device) 
        self.layer3 = nn.Linear(hidden_layer_neurons , 1).to(device) 
        self.gelu = nn.GELU()

    def forward(self,input_vec):
        input_vec_2 = self.gelu(self.layer1(input_vec))
        input_vec_3 = input_vec_2 + self.gelu(self.layer2(input_vec_2))
        return self.layer3(self.gelu(input_vec_3))

class MinMaxNorm_layers(nn.Module):
    def __init__(self,xmax,tmax,device):
        
        self.net_name = "minmaxlayers"
               
        super().__init__()

        # store normalization constants (not trainable)
        #qmax = self.boundary.qmax
        self.register_buffer("xmax", torch.tensor(xmax,dtype=torch.float32))
        self.register_buffer("xmin", torch.tensor(-xmax,dtype=torch.float32))

        self.register_buffer("tmax", torch.tensor(tmax,dtype=torch.float32))

        self.device = device

    def x_minmax_norm(self,x):
        xnorm = (x - self.xmin) / (self.xmax - self.xmin + 1e-8)
        return xnorm.to(self.device)

    def t_minmax_norm(self,x):
        xnorm = x / self.tmax
        return xnorm.to(self.device)

#@torch.compile
class NeuralNetwork(nn.Module):
    def __init__(self,boundary,device=None,nn_params=None):
        
        super().__init__()
  
        self.net_name = "du"
        
        if device=="cpu":
            self.device=device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                print("No GPU available. Training on CPU")
                self.device = "cpu"
        
        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = du_network_params()
        
        #params
        self.boundary = boundary

        self.flatten = nn.Flatten()
        
        self.FeedForwardLayers = NeuralNetwork_layers(self.device, self.params["duhiddenlayerneurons"])
        
        self.FourierFeatures = FourierFeatures(1,2*self.params["duhiddenlayerneurons"],scale=1,device=self.device)
        
        self.MinMaxNorms = MinMaxNorm_layers(xmax=self.boundary.norm_qmax,tmax=self.boundary.params["Tf"], device=self.device)
        
        self.checkpoint_path = fetch_checkpoint_path()
        
        if self.checkpoint_path is not None: #automatically loads a checkpoint from the command line
            try: 
                load_checkpoint(self,self.checkpoint_path)
                print(f"Checkpoint at {self.checkpoint_path} loaded successfully.")
            except BaseException as e:
                print(f"Checkpoint at {self.checkpoint_path} could not be loaded: {e}")
                raise

    def forward(self, input_vec):   #how the model should evaluate
        
        x = self.FeedForwardLayers.NNx(self.MinMaxNorms.x_minmax_norm(input_vec[...,[1]]))
        t = self.FourierFeatures(self.MinMaxNorms.t_minmax_norm(input_vec[...,[0]]))#

        tout = self.FeedForwardLayers.NNt(self.MinMaxNorms.t_minmax_norm(input_vec[...,[0]]))
        
        combined = self.FeedForwardLayers.layer_norm_combined(torch.cat((tout,x), dim=-1) )
        
        #concatenate the outputs and put them through the part 2
        out = self.FeedForwardLayers.NN1(combined) + self.FeedForwardLayers.NN2(t)*self.boundary.scorefinal(input_vec[...,[1]])
        
        return out

    def initialise_weights(self):   #sets all the weights away from zero.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight,gain=2)
                #if m.bias is not None:
                #    torch.nn.init.xavier_uniform_(m.bias, gain=1)

#@torch.compile
class RhoNeuralNetwork(nn.Module):
    """
    Neural Network object for parametrising the form of the probability distribution 
    """

    def __init__(self,nn_params=None):
        self.net_name = "rho"
   
        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = rho_network_params()
        
        super(RhoNeuralNetwork, self).__init__()
        self.NNx = nn.Sequential(
            nn.Linear(2, self.params["rhohiddenlayerneurons"] ),  #input x
            nn.Tanh(),
            nn.Linear(self.params["rhohiddenlayerneurons"] ,self.params["rhohiddenlayerneurons"] ),   #64 outputs
            nn.Tanh(),
            nn.Linear(self.params["rhohiddenlayerneurons"] ,self.params["rhohiddenlayerneurons"] ),   #64 outputs
            nn.Tanh(),
            nn.Linear(self.params["rhohiddenlayerneurons"] , 1),   #64 outputs
            #nn.ReLU(),
            )  

    def forward(self, x0, condition=""):   #how the model should evaluate
        
        if condition=="log": 
            return self.NNx(x0)
        else:
            return torch.exp(self.NNx(x0))

#@torch.compile
class MLP(nn.Module):
    """Defines a simple flexible model used for interpolating the benchmark solution 
    """

    def __init__(self,checkpointpath=None,net_name="pert"):
        """Initialise a neural network called mlp.
        This is a two layer feed forward network with 128 neurons per layer
        It can be used to conveniently interpolate functions from a csv

        args:
            checkpoint_path : str

            net_name : str
                descriptive identifier for loading saved checkpoints
        """

        #identifier to use in saving the checkpoint
        self.net_name = net_name

        self.params = du_network_params()
        neurons = 16 #self.params["duhiddenlayerneurons"]
        
        self.device = "cpu"

        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, neurons ),  #two inputs
            nn.Tanh(),
            nn.Linear(neurons , neurons),   #hidden layer
            nn.Tanh(),
            nn.Linear(neurons , 1)     #one output
        ).to(self.device)    

        self.MinMaxNorms = MinMaxNorm_layers(xmax=3,tmax=5,device=self.device)
        #self.initialize_weights()   

    def forward(self, input_vec):   #how the model should evaluate

        t = self.MinMaxNorms.t_minmax_norm(input_vec[...,[0]])
        x = self.MinMaxNorms.x_minmax_norm(input_vec[...,[1]])

        out = self.linear_stack(torch.cat((t,x), dim=-1))
        return out

    def initialize_weights(self):   #sets all the weights to zero.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,gain=4)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    from torchinfo import summary

    import pdb
    from kldescent.sde.boundary import UnderdampedGaussian
    boundary_cond = UnderdampedGaussian() #getattr(boundaries,run_params["boundary"])()
    
    net = NeuralNetwork(boundary_cond)

    summary(net, input_size=(1,2))