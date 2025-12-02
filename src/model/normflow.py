#tutorial from
#https://github.com/VincentStimper/normalizing-flows

# Import required packages
import torch
from torch import nn
import numpy as np

try:
    import normflows as nf
except ImportError:
    print("Package normflows has not been imported. Please activate the virtual environment before this package can be used.")
    raise ImportError
    pass

from tqdm import tqdm

from ..utils.parser import rho_network_params,fetch_checkpoint_path

class RhoLayer():
    def __init__(self,checkpointpath=None):
        
        self.one_linear = nn.Linear(1, 1)   

    def forward(self, x):   #how the model should evaluate
        out = self.linear_stack(x)
        return x + nn.RELU(out)

class RhoNormFlow():
    """model object for a neural network, trainable normalising flow that 
    learns a final distirbution
    """

    def __init__(self,nn_params=None):
        """Initialise the model """

        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = rho_network_params()
        
        # Define 2D Gaussian base distribution
        base = nf.distributions.base.DiagGaussian(2)

        # Construct flow model
        #check how to use a custom function for base?
        flows = self.make_flows(num_layers=4)
        self.model = nf.NormalizingFlow(base, flows)
        

    # Define list of flows
    def make_flows(self,num_layers=4):
        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([1, self.params["rhohiddenlayerneurons"],
                                        self.params["rhohiddenlayerneurons"], 2], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))
        
        return flows

class RhoPositionNormFlow(RhoNormFlow):
    """model object for a neural network, trainable normalising flow that 
    learns the final position distribution 
    """

    def __init__(self,nn_params=None):
        """Initialise the model 
        Initialise from the normalising flow learning the """


        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = rho_network_params()
        
        # Define Gaussian base distribution
        base = nf.distributions.DiagGaussian(1)
        self.model = nf.NormalizingFlow(base, flows=self.make_flows())#self.make_flow()

    # Define list of flows
    def make_flows(self,num_layers=4):
                
        latent_size = 1
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
        flows = []
        for i in range(num_layers):
            s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
            t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
            if i % 2 == 0:
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
            flows += [nf.flows.ActNorm(latent_size)]
        return flows

##training
class TrainNormFlow():
    """
    Class to store training loop and parameters for training a normalising flow
    """

    def __init__(self,model,boundaryconditions):
        """
        Initialise training parameters for trainign a normalising flow
        args:   
            model: 
                normalising flow model (neural net with planar layers)
            boundaryconditions:
                boundary conditions
        """

        self.model = model
        self.boundary = boundaryconditions

        self.max_iter = 400 #maximum number of iterations /training steps
        self.minibatch_samples = 128 #how many samples of final distribution to take for each training iteration
        #self.show_iter = 20 #at which iteration to plot intermediate results
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)

    def train_loop(self,integrator,x,x0):
        
        loss_hist = np.array([])

        for it in tqdm(range(self.max_iter)):
            
            #generate random indexes for the minibatch 
            idx = torch.randint(low=0,high=x.shape[0],size=(self.minibatch_samples,))

            self.train_step(x[idx,:],it,loss_hist)

    def train_step(self,x,it,loss_hist):
        
        self.optimizer.zero_grad()
           
        # Compute loss
        loss = self.model.forward_kld(x) 
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self.optimizer.step()
            
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

class TrainPositionNormFlow(TrainNormFlow):
    """
    Class to store training loop and parameters for training a normalising flow
    """

    def __init__(self,model,boundaryconditions):
        """
        Initialise training parameters for trainign a normalising flow to learn a one-dimensional distribution 
        args:   
            model: 
                normalising flow model (neural net with planar layers)
            boundaryconditions:
                boundary conditions
        """

        #initialise params as TrainNormFlow
        TrainNormFlow.__init__(self,model,boundaryconditions)
        
    def train_loop(self,integrator,x,x0):
        
        loss_hist = np.array([])

        for it in tqdm(range(self.max_iter)):
            
            #generate random indexes for the minibatch 
            idx = torch.randint(low=0,high=x.shape[0],size=(self.minibatch_samples,))

            self.train_step(x[idx],it,loss_hist)


    def train_step(self,x,it,loss_hist):
        
        self.optimizer.zero_grad()
           
        # Compute loss
        loss = self.model.forward_kld(x) 
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self.optimizer.step()
            
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

if __name__=="__main__":

    #initialise initial and final conditions
    boundaryconditions = UnderdampedGaussian()