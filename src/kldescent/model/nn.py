import math

import torch
from torch import nn

from kldescent.utils.parser import du_network_params,rho_network_params,fetch_checkpoint_path

def load_checkpoint(net,checkpointpath):
    """Handles loading neural network parameters from checkpoint, and uses error handling to consolidate mismatches by raising correct error 
    message. The error is then handled if necessary or runtime interrupted if it should not be raised.

    If checkpoint can by loaded, the network parameters will be loaded to the instance 
    and mapped to the net's device. 

    If no checkpoint file is found (FileNotFoundError)
    If checkpoint file format is incompatible (RuntimeError), e.g. caused by different number of neurons in instance and checkpoint 
    If checkpoint file format does not have correct attributes (AttributeError) e.g. Neural network architecture changed between runs.
    BaseException is raised. This may be handled in logic at the call sites of load_checkpoint, depending on use-case.     

    args: 
    net : instance
        instance of neural network. The architecture of the network should match
    checkpointpath: str
        location of checkpoint. Must be provided as full filepath from current working directory. 
    """
    
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
    """Holds the layers for Fourier feature encoding. 
    
    Used for time-embedding of the time variable. """
    
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
    """Defines the feed-forward layers of the neural network architecture. """

    def __init__(self,d,device,hidden_layer_neurons):
        
        self.net_name = "du_layers_only"
        self.d = d 
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
            nn.Linear(d,hidden_layer_neurons ),  #input x
            nn.GELU(),
            nn.Linear(hidden_layer_neurons, hidden_layer_neurons),   #64 outputs
            ).to(device)   
        self.NN1 = NN2(device,hidden_layer_neurons).to(device) 
        self.NN2 = NN2(device,hidden_layer_neurons).to(device) 
        self.layer_norm_combined = nn.LayerNorm(2*hidden_layer_neurons).to(device)

class NN2(nn.Module):
    """One three layer feed forward module of the neural network architecture."""

    def __init__(self,device,hidden_layer_neurons):
        super().__init__()
        self.layer1 = nn.Linear(2*hidden_layer_neurons , hidden_layer_neurons).to(device) 
        self.layer2 = nn.Linear(hidden_layer_neurons , hidden_layer_neurons).to(device) 
        self.layer3 = nn.Linear(hidden_layer_neurons , 1).to(device) 
        self.gelu = nn.GELU()
        #self.zero_init_anchor(self.layer3)

    def forward(self,input_vec):
        input_vec_2 = self.gelu(self.layer1(input_vec))
        input_vec_3 = input_vec_2 + self.gelu(self.layer2(input_vec_2))
        return self.layer3(self.gelu(input_vec_3))



class MinMaxNorm_layers(nn.Module):
    """class containing normalisation layers for the neural network
    not involved in gradient descent as use min-maxes defined by the boundary conditions. """
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
    r"""Class holding the complete neural network architecture. 
    
    Branches:

    $$
    T(\hat{t}) = \text{NNt}(\hat{t}) \in \mathbb{R}^H
    $$
    with NNt a feed-forward network instance from NeuralNetwork_layers

    $$
    \text{NNt}: \hat{t} \xrightarrow{L_1,\text{GELU}} \xrightarrow{L_H,\text{GELU}} \mathbb{R}^H
    $$

    $$
    X(\hat{x}) = \text{NNx}(\hat{x}) \in \mathbb{R}^H
    $$
    with NNx a feed-forward network instance from NeuralNetwork_layers
    $$
    \text{NNx}: \hat{x} \xrightarrow{L_d,\text{GELU}} \xrightarrow{L_H} \mathbb{R}^H
    $$
    Time variable uses Fourier feature positional embedding instance from FourierFeatures class
    $$
    F(\hat{t}) = \text{FourierFeatures}(\hat{t}) = \bigl[\sin(2\pi B\hat{t}),, \cos(2\pi B\hat{t})\bigr] \in \mathbb{R}^{2H}
    $$

    where $B \in \mathbb{R}^{H}$ is a trainable weight vector (initialised $\mathcal{N}(0,1)$).

    The two variables with a normalisation layer:

    $$
    c = \text{LayerNorm}\bigl([T(\hat{t}),, X(\hat{x})]\bigr) \in \mathbb{R}^{2H}
    $$

    Output (two residual paths through NN2):

    Each NN2 block acts as:

    $$
    \text{NN2}(v) = L_1\!\left(\text{GELU}\!\left(v_2 + \text{GELU}(L_2(v_2))\right)\right), \quad v_2 = \text{GELU}(L_3(v))
    $$

    with $v \in \mathbb{R}^{2H} \to \mathbb{R}^H \to \mathbb{R}^H \to \mathbb{R}^1$.

    Final output:

    $$
    \boxed{\mathbb{U}(t,x) = \text{NN2}1(c) + \text{NN2}2\!\bigl(F(\hat{t})\bigr) \cdot \nabla{x}\log\rho\text{final}(x)}
    $$

    The second term anchors the network to the score of the target distribution at all times via boundary.scorefinal.
    """
    
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
        self.d = self.boundary.d #dimension of position space

        self.flatten = nn.Flatten()
        
        self.FeedForwardLayers = NeuralNetwork_layers(self.d,self.device, self.params["duhiddenlayerneurons"])
        
        scale = 0.3 * self.boundary.params["Tf"] * (1 + 4 * self.boundary.params["epsilon"]**2) ** 0.5
        
        self.FourierFeatures = FourierFeatures(1,2*self.params["duhiddenlayerneurons"],scale=scale,device=self.device)
        
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
        
        x = self.FeedForwardLayers.NNx(self.MinMaxNorms.x_minmax_norm(input_vec[...,1:]))
        t = self.FourierFeatures(self.MinMaxNorms.t_minmax_norm(input_vec[...,[0]]))#

        tout = self.FeedForwardLayers.NNt(self.MinMaxNorms.t_minmax_norm(input_vec[...,[0]]))
        
        combined = self.FeedForwardLayers.layer_norm_combined(torch.cat((tout,x), dim=-1) )
        
        #concatenate the outputs and put them through the part 2
        out = self.FeedForwardLayers.NN1(combined) + self.FeedForwardLayers.NN2(t)*self.boundary.scorefinal(input_vec[...,1:])
        
        return out

    def initialise_weights(self):   #sets all the weights away from zero.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight,gain=2)
                #if m.bias is not None:
                #    torch.nn.init.xavier_uniform_(m.bias, gain=1)

    def grad_x_analytic(self, input_vec):
        """
        THIS FUNCTION MUST BE PROPERLY CHECKED BEFORE USE! PLACEHOLDER ONLY
        ####
        DO NOT USE!!!
        ####

        Gradient of output w.r.t. x = input_vec[..., 1:].
        Analytic gradient of the neural network NeuralNet with respect to the spatial input.
         
        In this way, it avoids calling autograd when accumulating the loss.
        
        Note: incompatible with autograd as is. Use for inference only.
        """
        import torch.nn.functional as F

        fl = self.FeedForwardLayers
        mm = self.MinMaxNorms
        H  = self.params["duhiddenlayerneurons"]

        raw_x = input_vec[..., 1:]
        raw_t = input_vec[..., [0]]

        def gelu_prime(z):
            return (0.5 * (1.0 + torch.erf(z / math.sqrt(2)))
                    + z * torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi))

        # ── Forward (cache activations needed for backward) ──────────────────

        scale_x = 1.0 / (mm.xmax - mm.xmin + 1e-8)
        x_hat   = mm.x_minmax_norm(raw_x)
        a_x1    = F.linear(x_hat, fl.NNx[0].weight, fl.NNx[0].bias)   # [..., H]
        h_x1    = F.gelu(a_x1)
        X       = F.linear(h_x1, fl.NNx[2].weight, fl.NNx[2].bias)    # [..., H]

        t_hat = mm.t_minmax_norm(raw_t)
        a_t1  = F.linear(t_hat, fl.NNt[0].weight, fl.NNt[0].bias)
        h_t1  = F.gelu(a_t1)
        a_t2  = F.linear(h_t1, fl.NNt[2].weight, fl.NNt[2].bias)
        T     = F.gelu(a_t2)                                            # [..., H]

        v     = torch.cat([T, X], dim=-1)                               # [..., 2H]
        mu    = v.mean(dim=-1, keepdim=True)
        sigma = (v.var(dim=-1, keepdim=True, unbiased=False)
                + fl.layer_norm_combined.eps).sqrt()
        v_hat = (v - mu) / sigma
        c     = fl.layer_norm_combined.weight * v_hat + fl.layer_norm_combined.bias

        nn1 = fl.NN1
        a1  = F.linear(c,  nn1.layer1.weight, nn1.layer1.bias)         # [..., H]
        h1  = F.gelu(a1)
        a2  = F.linear(h1, nn1.layer2.weight, nn1.layer2.bias)         # [..., H]
        h2  = h1 + F.gelu(a2)                                           # [..., H]

        b_t = fl.NN2(self.FourierFeatures(t_hat))                       # [..., 1]

        # ── Backward through NN1 (seed = 1) ──────────────────────────────────

        g_h2 = gelu_prime(h2) * nn1.layer3.weight[0]                   # [..., H]
        g_h1 = g_h2 + (gelu_prime(a2) * g_h2) @ nn1.layer2.weight     # [..., H]
        g_c  = (gelu_prime(a1) * g_h1) @ nn1.layer1.weight             # [..., 2H]

        # ── Backward through LayerNorm ────────────────────────────────────────

        g_vhat = fl.layer_norm_combined.weight * g_c
        g_v    = (g_vhat
                - g_vhat.mean(dim=-1, keepdim=True)
                - v_hat * (g_vhat * v_hat).mean(dim=-1, keepdim=True)) / sigma
        g_X    = g_v[..., H:]                                           # [..., H]

        # ── Backward through NNx ──────────────────────────────────────────────

        g_hx1  = g_X @ fl.NNx[2].weight                                # [..., H]
        g_ax1  = gelu_prime(a_x1) * g_hx1                              # [..., H]
        g_xhat = g_ax1 @ fl.NNx[0].weight                              # [..., d]
        g_x    = g_xhat * scale_x                                       # [..., d]

        # ── Second term: b(t) * ∇_x scorefinal(x) ────────────────────────────

        g_x = g_x + b_t * self.boundary.grad_x_scorefinal(raw_x)

        return g_x


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
        neurons = self.params["duhiddenlayerneurons"]
        
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

    from kldescent.sde.boundary import UnderdampedGaussian
    boundary_cond = UnderdampedGaussian() #getattr(boundaries,run_params["boundary"])()
    
    net = NeuralNetwork(boundary_cond)

    summary(net, input_size=(1,2))