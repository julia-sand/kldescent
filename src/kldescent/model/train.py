import torch
#from torch import nn as torchnn
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from kldescent.sde.girsanovpdf import compute_pdf


class Train():
    """
    Defines the parameters of the training process and the training steps
    for the neural network learning the non-perturbative drift.
    """

    def __init__(self, net, integrator):

        #set defaults of training process
        self.lr = 0.0005 #2.5e-4
        #0.0005 #initial learning rate
        self.integrator = integrator
        self.net = net

        self.gamma = 0.1 #rate of learning rate decrease.

        self.clip = 0.5 #gradient clipping

        self.optimiser_feedforward = AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        self.optimiser_fourier = AdamW(self.net.FourierFeatures.parameters(), lr=0.01)

        self.milestone = 200 #how often to decay learning rate
        self.start_decay = 20 #when to start decaying learning rate 
        
        #COSINE ANNEALING LR SCHEDULE https://medium.com/@utkrisht14/cosine-learning-rate-schedulers-in-pytorch-486d8717d541
        self.scheduler1 = CosineAnnealingLR(self.optimiser_fourier, 
                                        T_max=100, eta_min=1e-10)
        #self.scheduler = ExponentialLR(self.optimiser, gamma=self.gamma) #Decays the learning rate of each parameter group by gamma every ""step_size"".
        self.scheduler2 = ReduceLROnPlateau(self.optimiser_feedforward,
                                            mode="min",
                                            factor=0.9,
                                            patience=1, #how many steps to wait before 
                                            eps=1e-8,
                                            cooldown=3) #Decays the learning rate based on a passed metric
        
    def train_step(self):  
        
        self.net.train() #set neural net into train mode
        
        #reset gradients 
        self.optimiser_feedforward.zero_grad()
        self.optimiser_fourier.zero_grad()

        Lhat = self.integrator.loss_fn()

        if torch.isfinite(Lhat):
            Lhat.backward()
            clip_grad_norm_(self.net.FeedForwardLayers.parameters(), 0.5)
            clip_grad_norm_(self.net.FourierFeatures.parameters(), 0.5)

            self.optimiser_feedforward.step()
            self.optimiser_fourier.step()
            
        else: 
            print("Skipping step due to NaN/Inf loss:", Lhat.item())

            #reset gradients 
            self.optimiser_feedforward.zero_grad()
            self.optimiser_fourier.zero_grad()            
            
        
        return Lhat.detach().cpu().numpy()

class TrainRho():
    """
    Defines the training loop for the loss function 
    for learning boundary conditions
    
    overwrite any training params here
    
    args
        rhonet: neural network used to approximate the final distribtuion
        integrator: class that holds integration functions for the fokker planck
    """

    def __init__(self, rhonet):#, integrator):

        self.rhonet = rhonet 

        self.lr = 0.0001 #initial learning rate
        #self.integrator = integrator

        self.gamma = 0.1 #rate of learning rate decrease.

        self.clip = 0.01 #gradient clipping

        self.rho_epochs = 3000 #total number of training epochs
        self.milestone = 512 #how often to decay learning rate
        self.rho_batch = 128 #size of batch 

        self.loss = torch.nn.MSELoss()

    @torch.no_grad
    def generate_training_data(self,x0,integrator):
        
        with torch.no_grad():
            yobs = torch.zeros_like(x0[...,[0]])

            yobs = compute_pdf(x0,integrator.net,
                                integrator.boundary.sigma0,
                                integrator.boundary.params["epsilon"],
                                integrator.boundary.p_initial,
                                integrator.boundary.params["Tf"],
                                mc_samples=1)
            #print(yobs.max())
            #print(torch.isnan(yobs).any())                    
            return yobs

    def train_step(self,optimiser,integrator):  

        #reset gradients 
        optimiser.zero_grad()

        #initialise x values 
        x0 = integrator.boundary.generate_samples(self.rho_batch,mode="initial") 

        yobs = self.generate_training_data(x0.clone().detach(),integrator)

        ypred = self.rhonet(x0)

        Lhat = self.loss(ypred,yobs)

        # Do backprop and optimiser step
        if torch.isfinite(Lhat):
            ##training##
            Lhat.backward()

            clip_grad_norm_(self.rhonet.parameters(), self.clip)
             
            optimiser.step()
        else: 
            print("Skipping training skep due to NaN loss.")
            optimiser.zero_grad()

        #reset gradients 
        optimiser.zero_grad()
          
        #TODO loss function may be nan when the initialisation of the drift 
        #network is too large. Fix this in future by adding a conditional warmstart
        #to DUNET whenever terminal cost is GirsanovNet

        return 


    def train_loop(self, integrator, rho_epochs=None):
        """
        loop for training the rho neural net
        args:
            epochs:
                number of epochs to train the network between updates to the drift net
        """

        if rho_epochs==None:
            rho_epochs=self.rho_epochs
        
        #reset optimiser and scheduler for each train loop
        optimiser = Adam(self.rhonet.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimiser, 
                            T_max=100, eta_min=1e-10)
                            #ExponentialLR(optimiser, gamma=self.gamma)
        self.rhonet.train() #set neural net into train mode
        
        with tqdm(total=int(round(rho_epochs/self.milestone)), bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
                postfix=["Batch", {"value": 0}]) as t:
            for rho_epoch in range(int(round(rho_epochs/self.milestone))):    
            
                for __ in range(self.milestone):
                    self.train_step(optimiser,integrator) #repeat training step
            
                scheduler.step() #decrease learning rate
                
                t.postfix[1]["value"] = rho_epoch
                t.update()

        self.rhonet.eval() #return neural net into eval mode

if __name__=="__main__":
    Pretrain(folder="src/refsolution/UnderdampedHalfNet/", path_to_pretrain="benchmark.csv").get_pretraining_data()
