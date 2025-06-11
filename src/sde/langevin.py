import numpy as np 
import torch

import src.sde.kl as kl
from ..utils.parser import integrator_params

#import pdb

class SDEOverdamped():
    def __init__(self,
                    boundary_data,
                    logger=None,
                    device="cpu"):
        self.boundary = boundary_data
        self.logger = logger

        #get params 
        self.sde_params = integrator_params()
        self.batch_samples = self.sde_params.batchsamples # number of samples in each step 
        self.t_steps = self.sde_params.tsteps #number of timesteps in discretisation
        #self.Tf = self.boundary.Tf
        self.g = self.boundary.g
        if self.logger is not None:
            self.logger.log_args(self.sde_params) #add to log


    def generate_time_grid(self):

        #samples the interval [0,Tf] and returns sorted set of values to be used in integration

        return np.concatenate([[0],
                                np.sort(np.random.uniform(0, self.boundary.Tf, self.t_steps-1)),
                                [self.boundary.Tf]])


    def testing_time_grid(self,final_time=None):

        if final_time==None:
            final_time = self.boundary.Tf
            
        #get an equally spaced time grid which can be used with the reference solution
        time_grid = self.boundary.get_reference_timegrid()
        if time_grid is not None:
            return time_grid
        else:
            return np.linspace(0,final_time,self.t_steps)

    def generate_innovation(self,dt,size):
        #generate a step of brownian motion
        return np.sqrt(dt)*torch.randn((size,1),requires_grad=False,dtype=torch.float32)

    def simulation_free(self,net):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network
        ====

        """
        
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.batch_samples,"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)

        #set up a,abar,c,cbar
        a = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        abar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        dw = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)

        #pdb.set_trace()
        #self.generate_time_grid()
        for i in range(0,self.t_steps-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,self.batch_samples)
           
            du = net.forward(torch.column_stack((t_vec[i]*torch.ones(self.batch_samples,dtype=torch.float32)
                                                    ,xinit[:,0,None])))
                    
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True) ** 2) 
            
            c = c + (du*dw)
            du_bar = du.detach()#.numpy() #untracked copy

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                
            xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])

        #compute value of Lhat
        #see equation (10) and (11) in ref. 2410.05163
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + self.boundary.gfun(xinit)) * c))
        
        #free up memory again
        del xinit 
        del du
        del du_bar
        del t_vec
        del dw

        a = None
        abar = None
        bbar = None
        c = None

        return Lhat
    
    #@profile
    def compute_metrics(self,net,epoch,lout):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network
        ====

        """


        net.eval()
        t_vec = self.testing_time_grid()
        testing_samples = 1000 #reduced number of samples in computing the loss metrics

        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.tensor(self.boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)
        xinit_ref = torch.clone(xinit)

        kldyn = torch.zeros((testing_samples,1),requires_grad=False,dtype=torch.float32)
        #
        #self.generate_time_grid()
        for i in range(0,len(t_vec)-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,testing_samples)


            du_ref = torch.tensor(self.boundary.du_ref(xinit_ref[:,0,None],t_vec[i]),
                                    requires_grad=False,
                                    dtype=torch.float32)

            with torch.no_grad():

                du_bar = net(torch.column_stack((t_vec[i]*torch.ones(testing_samples,requires_grad=False,dtype=torch.float32)
                                                        ,xinit[:,0,None]))).detach()
                kldyn = kldyn + dt*(torch.linalg.norm(torch.sub(du_bar, du_ref), axis=1,keepdim=True)) 
                #torch.randn((testing_samples,1),requires_grad=False,dtype=torch.float32)


                xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])
        
            xinit_ref = self.boundary.euler_maruyama_step(xinit_ref,du_ref,dt,dw,t_vec[i])
        
        try:
            kl_static = kl.kl_div(xinit,xinit_ref)
        except ValueError:
            kl_static = 0
            print("Value Error exception ignored. This means that the current value of the \n \
            neural network drift produces NaN values. ")
        
        if self.logger is not None:
            self.logger.log_epoch_data(net,
                                    epoch,
                                    lout,
                                    (self.boundary.beta*self.boundary.tau/(4*self.boundary.m))*torch.mean(kldyn).item(),
                                    kl_static)

        #free up the tensors
        del xinit 
        del xinit_ref
        #del du
        del du_bar
        del du_ref
        del kldyn


        return 


if __name__=="__main__":
    pass