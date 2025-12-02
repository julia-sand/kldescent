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


class CustomNormFlow(nf.NormalizingFlow):
    """Overwrite the base class to use custom probabilities
    """

    def __init__(self,):
        #inherit from parent
        nf.NormalizingFlow.__init__()

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q
    


    def forward_kld(self, x):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)


class RhoNormFlow():
    """model object for a neural network, trainable normalising flow that 
    learns a final distirbution
    """

    def __init__(self,nn_params=None,bridgetype="marginal"):
        """Initialise the model """

        if nn_params is not None:
            self.params = nn_params    
        else:
            self.params = rho_network_params()
        
        
        if bridgetype=="marginal":
            self.latent_size = 1 
        else:
            self.latent_size = 2 
        
        # Define 2D Gaussian base distribution
        base = nf.distributions.base.DiagGaussian(self.latent_size)

        # Construct flow model
        #check how to use a custom function for base?
        
        flows = self.make_flows(num_layers=4)
        self.model = nf.NormalizingFlow(base, flows=flows)#self.make_flow()
        
    def make_flows(self,num_layers=25):
                
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.latent_size)])
        flows = []
        for i in range(num_layers):
            s = nf.nets.MLP([self.latent_size, 64, 64, self.latent_size], init_zeros=True)
            t = nf.nets.MLP([self.latent_size, 64, 64, self.latent_size], init_zeros=True)
            if i % 2 == 0:
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
            flows += [nf.flows.ActNorm(self.latent_size)]
        
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
        self.minibatch_samples = 256 #how many samples of final distribution to take for each training iteration
        #self.show_iter = 20 #at which iteration to plot intermediate results
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3, weight_decay=1e-5)

    def train_loop(self,x,x0):
        
        loss_hist = np.array([])

        for it in tqdm(range(self.max_iter)):
            
            #generate random indexes for the minibatch 
            idx = torch.randint(low=0,high=x.shape[0],size=(self.minibatch_samples,))

            self.train_step(x[idx,:],x0[idx,:],it,loss_hist)

    def train_step(self,x,x0,it,loss_hist):
        
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