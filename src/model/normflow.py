#tutorial from
#https://github.com/VincentStimper/normalizing-flows

# Import required packages
import torch
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt

from tqdm import tqdm

from ..sde.boundary import UnderdampedHalfNet

#initialise initial and final conditions
boundaryconditions = UnderdampedHalfNet()

# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(2)

# Define list of flows
def make_flows(num_layers=4):
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 16, 16, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))
    return flows
        
# Construct flow model
#check how to use a custom function for base?
flows = make_flows(num_layers=4)
model = nf.NormalizingFlow(base, flows)

# Plot target distribution
grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))

def plot_prob(prob,xx,yy,fig_name="test"):
    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(f"{fig_name}.png")
    plt.close()
    return


##training
class Train_normflow():
    """
    Class to store training loop and parameters for training a normalising flow
    """

    def __init__(self,model):
        """
        Initialise training parameters for trainign a normalising flow
        args:   
            model: 
                normalising flow model (neural net with planar layers)
        """

        self.model = model

        self.max_iter = 400 #maximum number of iterations /training steps
        self.num_samples = 2000 #how many samples of final distribution to take for each training iteration
        self.show_iter = 20 #at which iteration to plot intermediate results
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)

    def train_loop(self,):
        
        loss_hist = np.array([])

        for it in tqdm(range(max_iter)):
            #get data here!!
            x = torch.tensor(boundaryconditions.generate_samples(num_samples,mode="final"),dtype=torch.float32)
            self.train_step(x,it,loss_hist)

        # Plot loss
        plt.figure(figsize=(10, 10))
        plt.plot(loss_hist, label='loss')
        plt.savefig("loss_training_example.png")
        plt.close()

    def train_step(self,x,it,loss_hist):
        
        optimizer.zero_grad()
           
        # Compute loss
        loss = model.forward_kld(x)
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

