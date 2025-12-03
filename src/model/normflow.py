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

    def __init__(self,boundary):
        nf.distributions.BaseDistribution.__init__(self)
        self.boundary = boundary #assign the boundary conditions
                
    def log_prob(self, z):
        """Calculate log probability of batch of samples

        Args:
          z: Batch of random variables to determine log probability for

        Returns:
          log probability for each batch element
        """
        
        logprobtemp = - self.boundary.Uinitial(z) - \
             torch.log(torch.ones(1,dtype=torch.float32)*self.boundary.pi_norm)
        
        return logprobtemp[:,0]
        
    def forward(self, num_samples=1, context=None):
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
        else:
            self.latent_size = 2 
        
        self.boundary = boundaryconditions
        
        # Define the base distribution 
        base = CustomBase(self.boundary)

        flows = self.make_flows(num_layers=50)
        self.model = nf.NormalizingFlow(base, flows=flows)#self.make_flow()
        
        #training parameters
        self.max_iter = 4000 #maximum number of iterations /training steps
        self.minibatch_samples = 128 #how many samples of final distribution to take for each training iteration
        #self.show_iter = 20 #at which iteration to plot intermediate results
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3, weight_decay=1e-5)

    def make_flows(self,num_layers=25):
                
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.latent_size)])
        flows = []
        for i in range(num_layers):
            s = nf.nets.MLP([self.latent_size, 64, self.latent_size], init_zeros=True)
            t = nf.nets.MLP([self.latent_size, 64, self.latent_size], init_zeros=True)
            if i % 2 == 0: #even layers
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else: #odd layers
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
            flows += [nf.flows.ActNorm(self.latent_size)]
        
        return flows

    def train_loop(self,x):
        
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
            #loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            #print(loss.to('cpu').data.numpy())

if __name__=="__main__":

    #initialise initial and final conditions
    boundaryconditions = UnderdampedGaussian()