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
        
    torch.save(net.state_dict(),loc) 

##SOURCE: https://github.com/miaotianyi 

class FourierFeatures(nn.Module):
    def __init__(self,in_features,out_features, scale, device):
        super().__init__()
       
        assert out_features % 2 == 0, "number of out_features must be even"
        self.device = device

        #initialise randn 
        self.B = scale*torch.randn(in_features, out_features//2).to(self.device)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        
        t_proj = 2 * math.pi * t @ self.B
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1).to(self.device)


class MyActivation(nn.Module): 
    def forward(self,x):
        return torch.sin(x)

#@torch.compile
class NeuralNetwork(nn.Module):
    def __init__(self,boundary,device=None,nn_params=None):
        
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

        super(NeuralNetwork, self).__init__()

        # store normalization constants (not trainable)
        #qmax = self.boundary.qmax
        self.register_buffer("xmax", torch.tensor(7))
        self.register_buffer("xmin", torch.tensor(-7))
        self.register_buffer("tmax", torch.tensor(5))
        
        self.flatten = nn.Flatten()
        #self.tencoding = TimeConder(1,self.hidden_layer_neurons) #fourier positional encoding goes here! 
        self.NNt = nn.Sequential(
            nn.Linear(1, self.params["duhiddenlayerneurons"] ),  #
            #nn.Linear(2*self.params["duhiddenlayerneurons"], self.params["duhiddenlayerneurons"] ),  #
            nn.GELU(), #MyActivation(),
            nn.Linear(self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),   #64 outputs
            #nn.GELU(), #MyActivation(),
            #nn.Linear(self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),   #64 outputs
            nn.ReLU(),  #must be positive
            ).to(self.device)
        self.NNx = nn.Sequential(
            nn.Linear(1, self.params["duhiddenlayerneurons"] ),  #input x
            nn.GELU(),
            nn.Linear(self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),   #64 outputs
            ).to(self.device)   
        self.NN1 = nn.Sequential(
            nn.Linear(2*self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            nn.GELU(),
            nn.Linear(self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),   #64 outputs
            nn.GELU(),
            nn.Linear(self.params["duhiddenlayerneurons"] , 1)).to(self.device) 
        self.NN2 = nn.Sequential(
            nn.Linear(2*self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),  #input 2x64 features: concatenated output of NNt and NNx t and x 
            nn.GELU(),
            nn.Linear(self.params["duhiddenlayerneurons"] , self.params["duhiddenlayerneurons"] ),   #64 outputs
            nn.GELU(),
            nn.Linear(self.params["duhiddenlayerneurons"] , 1),   #64 outputs
            nn.ReLU(), #must be positive
            ).to(self.device)
        self.FourierFeatures = FourierFeatures(1,2*self.params["duhiddenlayerneurons"],scale=0.5,device=self.device)
        
        self.layer_norm_combined = nn.LayerNorm(2*self.params["duhiddenlayerneurons"]).to(self.device)
        #self.layer_norm = nn.LayerNorm(self.params["duhiddenlayerneurons"])
        
        self.checkpoint_path = fetch_checkpoint_path()
        if self.checkpoint_path is not None: #automatically loads a checkpoint from the command line
            try: 
                load_checkpoint(self,self.checkpoint_path)
                print(f"Checkpoint at {self.checkpoint_path} loaded successfully.")
            except BaseException:
                #print(f"Checkpoint at {self.checkpoint_path} loaded successfully.")
                pass

    def x_minmax_norm(self,x):
        xnorm = (x - self.xmin) / (self.xmax - self.xmin + 1e-8)
        return xnorm.to(self.device)

    def t_minmax_norm(self,x):
        xnorm = x / self.tmax
        return xnorm.to(self.device)

    def forward(self, input_vec):   #how the model should evaluate
        
        #minmax norm on x
        #xnorm = 2 * (input_vec[...,[1]] - self.xmin) / (self.xmax - self.xmin + 1e-8) - 1

        #x = self.NNx(xnorm.to(self.device))
        #t = self.FourierFeatures(input_vec[...,[0]])#
        
        x = self.NNx(self.x_minmax_norm(input_vec[...,[1]]))
        t = self.FourierFeatures(self.t_minmax_norm(input_vec[...,[0]]))#

        tout = self.NNt(self.t_minmax_norm(input_vec[...,[0]]))
        
        combined = self.layer_norm_combined(torch.cat((tout,x), dim=-1) )
        #concatenate the outputs and put them through the part 2
        out = self.NN1(combined) + self.NN2(t)*self.boundary.scorefinal(input_vec[...,[1]])
        
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
            nn.ReLU(),
            nn.Linear(self.params["rhohiddenlayerneurons"] ,self.params["rhohiddenlayerneurons"] ),   #64 outputs
            nn.ReLU(),
            nn.Linear(self.params["rhohiddenlayerneurons"] ,self.params["rhohiddenlayerneurons"] ),   #64 outputs
            nn.ReLU(),
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
    """Defines a simple flexible model
    used for simple regression problems. 
    """

    def __init__(self,checkpointpath=None,net_name="pert"):
        """Initialises a neural network called mlp.
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
        
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, neurons ),  #two inputs
            nn.GELU(),
            nn.Linear(neurons , neurons),   #hidden layer
            nn.GELU(),
            nn.Linear(neurons , neurons),   #hidden layer
            nn.GELU(),
            nn.Linear(neurons , neurons),   #hidden layer
            nn.GELU(),
            nn.Linear(neurons , 1)     #one output
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

    import pdb
    from kldescent.sde.boundary import UnderdampedGaussian
    boundary_cond = UnderdampedGaussian() #getattr(boundaries,run_params["boundary"])()
    
    net = NeuralNetwork(boundary_cond)

    summary(net, input_size=(1,2))