import numpy as np 
import torch

import src.sde.kl as kl
from ..utils.parser import integrator_params

class SDEBase():
    """
    Class for integrating the sde forward and computing the loss function 
    """

    def __init__(self,boundary_data,
                    sde_params=None):

        #self.rhonet = rhonet #to be replaced with Terminal cost class
        self.boundary = boundary_data

        #get params 
        if sde_params is not None:
            self.sde_params = sde_params #take params from given yaml
        else:
            self.sde_params = integrator_params() #parse command line

    def generate_time_grid(self):
        """
        Samples the interval [0,Tf] and returns sorted set of values to be used in integration
        """

        return np.concatenate([[0],
                                np.sort(np.random.uniform(0, self.boundary.params["Tf"], self.sde_params["tsteps"]-1)),
                                [self.boundary.params["Tf"]]])

    def generate_innovation(self,dt,size):
        #generate a step of brownian motion
        return np.sqrt(dt)*torch.randn((size,1),requires_grad=False,dtype=torch.float32)

    def testing_time_grid(self,final_time=None):

        if final_time==None:
            final_time = self.boundary.params["Tf"]
        
        #get an equally spaced time grid which can be used with the reference solution
        time_grid = self.boundary.get_reference_timegrid()
        if time_grid is not None:
            return time_grid
        else:
            return np.linspace(0,final_time,self.sde_params["tsteps"])


class SDE(SDEBase):

    def __init__(self,
                    boundary_data,
                    terminal_cost=None, 
                    net=None,
                    sde_params=None,
                    #logger=None,
                    device="cpu"):
        
        SDEBase.__init__(self,boundary_data=boundary_data,sde_params=sde_params)

        self.TerminalCost = terminal_cost
        self.net = net 
        
    def loss_fn(self):
        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network for parametrising the drift
        ====
        """
        
        #self.net.train()
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)
        #xclone = xinit.clone().detach()
        
        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        abar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        dw = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)

        #pdb.set_trace()
        #self.generate_time_grid()
        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,self.sde_params["batchsamples"])
           
            du = self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                                                    ,xinit[...,[0]])))
                    
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True) ** 2) 
            
            c = c + (du*dw)
            du_bar = du.detach()#.numpy() #untracked copy

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                
            xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])

        with torch.no_grad():
            #if (self.rhonet is not None):
            #    xtemp = xinit.clone().detach()
            #    gvals = self.boundary.gfun(xtemp.numpy(),self.rhonet(xtemp,condition=1).numpy()) #self.rhonet(xclone))
            #else:
            #    gvals = self.boundary.gfun(xinit.numpy())
            #TODO update class structure so that we have
            gvals = self.TerminalCost.gfun(xinit.clone().detach())
                
        #compute value of Lhat
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + gvals) * c))
        
        return Lhat
    
    #@profile
    def compute_metrics(self):
        """
        Computes sample evolutions of the dynamics and saves the metrics for 
        recording the training process

        ====
        Parameters
            net: class
                neural network for estimating the 

        Returns:

        ====
        """

        self.net.eval()
        t_vec = self.testing_time_grid()
        testing_samples = self.sde_params["testsamples"] #4000 #reduced number of samples in computing the loss metrics

        #initialise x values as samples from assigned initial distribution
        xinit = torch.tensor(self.boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)
        xinit_ref = torch.clone(xinit)

        #preallocate storing kl divergence 
        kldyn = torch.zeros((testing_samples,1),requires_grad=False,dtype=torch.float32)

        for i in range(0,len(t_vec)-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,testing_samples)

            du_ref = torch.tensor(self.boundary.du_ref(xinit_ref[:,0,None],t_vec[i]),
                                    requires_grad=False,
                                    dtype=torch.float32)

            with torch.no_grad():

                du_bar = self.net(torch.column_stack((t_vec[i]*torch.ones((testing_samples,1),requires_grad=False,dtype=torch.float32)
                                                        ,xinit[...,[0]]))).detach()
                kldyn = kldyn + dt*(torch.linalg.norm(torch.sub(du_bar, du_ref), axis=1,keepdim=True)) 
                #torch.randn((testing_samples,1),requires_grad=False,dtype=torch.float32)


                xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])
        
            xinit_ref = self.boundary.euler_maruyama_step(xinit_ref,du_ref,dt,dw,t_vec[i])
        
        try:
            kl_static = 0#kl.kl_div(xinit,xinit_ref)
        except ValueError:
            kl_static = 0
            print("Value Error exception ignored. This means that the current value of the \n \
            neural network drift produces NaN values. ")
        mean0,var0 = torch.mean(xinit[...,0]).item(),torch.var(xinit[...,1]).item()

        return torch.mean(kldyn).item(), kl_static, mean0, var0 
    
    def update_ref_statistics(self,testing_samples=None):
        """
        computes estimates of the metrics to update the g function 

        ====
        Parameters
        - net: neural network
        Returns
            xinit
        ====

        """

        #if self.boundary.__class__.__name__ in ["UnderdampedHalfGaussian","UnderdampedHalfGaussBridge"]:
        if testing_samples == None: #default to the integrator 
            testing_samples = self.sde_params["testsamples"] 

        self.net.eval()
        
        with torch.no_grad():

            t_vec = torch.linspace(0,self.boundary.params["Tf"],self.sde_params["tsteps"])#self.testing_time_grid()

            xinit = torch.tensor(self.boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)

            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
                
                dw = self.generate_innovation(dt,testing_samples)
                #xdata = torch.column_stack(())
                du_bar = self.net(torch.column_stack((t_vec[i]*torch.ones((testing_samples,1),requires_grad=False,dtype=torch.float32)
                                                        ,xinit[...,[0]])))
                
                xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])
        
            #self.boundary.set_ref_statistics(torch.nanmean(xinit[...,0]).item(), torch.var(xinit[...,0]).item())
            #if self.boundary.__class__.__name__== "UnderdampedHalfGaussBridge":
            #    self.boundary.save_histogram(xinit.detach().numpy())
            #self.TerminalCost.update_cost_params(xinit.detach().numpy())
        
        return xinit

    def save_cumulants(self,save_array,temp):
      """
      appends the computed cumulants to a new array
      """
      
      return np.append(save_array,np.array([[np.nanmean(temp[...,0]),
                                              np.nanmean(temp[...,1]),
                                              np.nanvar(temp[...,0],ddof=1),
                                              np.nanvar(temp[...,1],ddof=1),
                                              np.correlate(temp[...,0],temp[...,1])[0]/(len(temp[...,0]))]]),axis=0)#cross corr

    def compute_sde(self,
                        final_time):
        """
        Computes the SDE while saving cumulants 

        ====
        args:
          final_time: when to stop the integration for intermediate time plots
        ====
        """
            
        #pdb.set_trace()
        t_vec = self.boundary.get_reference_timegrid(final_time)

        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["testsamples"],"initial"),requires_grad=False,dtype=torch.float32)
        xref = torch.clone(xinit)

        #save the cumulants of the evolutions
        cumulants_nn = self.save_cumulants(np.empty((0,5)),xinit.numpy())
        cumulants_ref = self.save_cumulants(np.empty((0,5)),xref.numpy())

        with torch.no_grad():

            #self.generate_time_grid()
            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
                
                dw = self.generate_innovation(dt,self.sde_params["testsamples"])
                du = self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["testsamples"],1),dtype=torch.float32)
                                                        ,xinit[...,[0]])))
                
                xinit = self.boundary.euler_maruyama_step(xinit,du,dt,dw,t_vec[i])
                xref = self.boundary.euler_maruyama_step(xref,self.boundary.du_ref(xref[:,[0]],t_vec[i]),dt,dw,t_vec[i])
                cumulants_nn = self.save_cumulants(cumulants_nn,xinit.numpy())
                cumulants_ref = self.save_cumulants(cumulants_ref,xref.numpy())
        
        return t_vec, xinit.numpy(), xref.numpy(), cumulants_nn, cumulants_ref

if __name__=="__main__":
    from src.model import nn
    from src.sde.boundary import UnderdampedHalfGaussian
    import pdb
    from src.utils.parser import integrator_params

    boundaryconditions = UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    integrator = SDE(boundaryconditions)

    integrator.update_ref_statistics(10,debug=True)

