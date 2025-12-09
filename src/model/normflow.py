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


class CustomBase(nf.distributions.BaseDistribution):

    def __init__(self,boundary,latent_size):
        nf.distributions.BaseDistribution.__init__(self)
        self.boundary = boundary #assign the boundary conditions
        self.latent_size = latent_size

    def log_prob(self, z):
        """Calculate log probability of batch of samples using the assigned initial 
        probability density

        Args:
          z: Batch of random variables to determine log probability for

        Returns:
          log probability for each batch element
        """
        
        if self.latent_size == 1:    
            logprob = - self.boundary.Uinitial(z) - \
             torch.log(self.boundary.pi_norm*torch.ones(1,dtype=torch.float32))
            #return logprob.flatten()
        else:
            logprob = - self.boundary.Uinitial(z[:,0]) - (z[:,1]-1)**2/2 \
             - 0.5*(torch.log(self.boundary.pi_norm*torch.ones(1,dtype=torch.float32)) + torch.log(4*torch.acos(torch.zeros(1,dtype=torch.float32)))) #hacky way to get pi into pytorch
            
        return logprob.flatten()
            
    def forward(self, num_samples=1, context=None):
        
        if self.latent_size == 1:    
            z = torch.tensor(self.boundary.generate_samples_q(num_samples,mode="initial"),dtype=torch.float32)
            return z, self.log_prob(z)
        else:
            z = torch.tensor(self.boundary.generate_samples(num_samples,mode="initial"),dtype=torch.float32)
            return z, self.log_prob(z)

class RhoNormFlow():
    """model object for a neural network, trainable normalising flow that 
    learns a final distirbution
    """

    def __init__(self,boundaryconditions,nn_params=None,bridgetype="marginal"):
        """Initialise the model """

        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = rho_network_params()
        
        if bridgetype=="marginal":
            self.latent_size = 1 
            flows = self.make_flows_1d(num_layers=5)

        else:
            self.latent_size = 2 
            flows = self.make_flows_2d(num_layers=50)

        self.boundary = boundaryconditions
        
        #training parameters
        self.max_iter = 1 #maximum number of iterations /training steps
        self.minibatch_samples = 128 #how many samples of final distribution to take for each training iteration
        
        # Define the base distribution 
        base = nf.distributions.DiagGaussian(1)#CustomBase(self.boundary,self.latent_size)
        #base = nf.distributions.DiagGaussian(1)
        self.model = nf.NormalizingFlow(base, flows=flows)
        
        #set optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)

    def make_flows_1d(self,num_layers=1):
                
        #b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.latent_size)])
        flows = []
        for i in range(num_layers):

            flows.append(nf.flows.Planar(shape=self.latent_size, act="leaky_relu"))

        return flows

    def make_flows_2d(self,num_layers=1):
                
        #b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.latent_size)])
        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))
            
        return flows 
        
    def train_loop(self,x):
        
        loss_hist = np.array([])

        for it in tqdm(range(self.max_iter)):
            
            #generate random indexes for the minibatch 
            idx = torch.randint(low=0,high=x.shape[0],size=(self.minibatch_samples,))
            #print(x[idx,:].shape)
            self.train_step(x[idx,:],it,loss_hist)

    def train_step(self,x,it,loss_hist):
        
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.model.forward_kld(x) 
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self.optimizer.step()
            
        else:
            raise BaseException 
            # Log loss
            #loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
        #print(loss.to('cpu').data.numpy())

if __name__=="__main__":

    #initialise initial and final conditions
    boundaryconditions = UnderdampedGaussian()