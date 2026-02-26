"""Implement the integration of SDE processes"""

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

        self.boundary = boundary_data

        #get params 
        if sde_params is not None:
            self.sde_params = sde_params #take params from given yaml
        else:
            self.sde_params = integrator_params() #parse command line

    def generate_time_grid(self,Tf=None,tsteps=None):
        """
        Samples the interval [0,Tf] and returns sorted set of values to be used in integration
        """

        if Tf ==None:
            Tf = self.boundary.params["Tf"]
        if tsteps==None:
            tsteps= self.sde_params["tsteps"]
        
        #TODO : get self.sde_params["tsteps"] as a fraction of the total time
        
        return np.concatenate([[0],
                                np.sort(np.random.uniform(0, Tf, tsteps-1)),
                                [Tf]])

    def generate_innovation(self,dt,size):
        """generate a step of brownian motion"""
        return np.sqrt(dt)*torch.randn((size,1),requires_grad=False,dtype=torch.float32)

class SDE(SDEBase):

    def __init__(self,
                    boundary_data,
                    terminal_cost=None, 
                    net=None,
                    sde_params=None,
                    device="cpu"):
        
        SDEBase.__init__(self,boundary_data=boundary_data,sde_params=sde_params)

        self.TerminalCost = terminal_cost
        self.net = net 
        self.testing_samples =self.sde_params["testsamples"]
        
    @torch.no_grad()
    def integration_step(self,xinit,dt,t):
        
        dw = self.generate_innovation(dt,self.testing_samples)

        #compute du 
        du = self.net(torch.column_stack((t*torch.ones((self.testing_samples,1),requires_grad=False,dtype=torch.float32)
                                                            ,xinit[...,[0]]))).detach()
                    
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt)

        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
            
        return xinit, du 

    @torch.no_grad()
    def reference_integration_step(self,xinit,dt,t,reference_sol):
        
        dw = self.generate_innovation(dt,self.testing_samples)

        if reference_sol.duref_net is not None:
            du = reference_sol.duref_net(torch.column_stack((t_vec[i]*torch.ones((self.testing_samples,1)),xinit[...,[0]])))
        else: 
            du = torch.as_tensor(reference_sol.get_ref_drift_gaussian(t),dtype=torch.float32)*xinit[...,[0]]
            #torch.zeros_like(xinit_ref[:,0,None],dtype=torch.float32,requires_grad=False)

        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt)

        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
            
        return xinit, du 


    def loss_fn(self):
        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters

        ====
        """
        
        #self.net.train()
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)
        
        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        abar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        dw = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)

        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,self.sde_params["batchsamples"])
           
            #get neural network drift
            du = self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                                                    ,xinit[...,[0]])))
                    
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True) ** 2) 
            
            c = c + (du*dw)
            du_bar = du.detach() #copy without gradient 

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                
            xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])

        with torch.no_grad():
            #compute terminal cost
            gvals = self.TerminalCost.gfun(xinit.clone().detach())
                
        #compute value of Lhat
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + gvals) * c))
        
        return Lhat
    
    #@profile
    def compute_metrics(self,reference_sol=None):
        """
        Computes sample evolutions of the dynamics and saves the metrics for 
        recording the training process

        ====
        Parameters:
            reference_sol : class
                reference_solution 
        Returns:
            kldyn : float
                Dynamic KL divergence computed along trajectories. 
            kl_static : float
                Estimate of KL divergence between target final and reached final distribution
            mean0 : float
                Estimated sample mean of the final achieved distribution
            var0 : float 
                Estimated sample variance of the final achieved distribution
        ====
        """

        with torch.no_grad():

            self.net.eval()
                  
            if reference_sol.duref_net is not None:
                t_vec = self.generate_time_grid() #use random time grid
            else: 
                t_vec = reference_sol.get_underdamped_tgrid() #use fixed times from csv
                            
            #initialise x values as samples from assigned initial distribution
            xinit = torch.tensor(self.boundary.generate_samples(self.testing_samples,"initial"),requires_grad=False,dtype=torch.float32)
            xinit_ref = torch.clone(xinit)

            #preallocate storing kl divergence 
            kldyn = torch.zeros((self.testing_samples,1),requires_grad=False,dtype=torch.float32)
            
            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
            
                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i])
        
                xinit_ref = self.reference_integration_step(xinit_ref,dt,t_vec[i],reference_sol)
                
                kldyn = kldyn + dt*(du_bar**2)
            
        return 0.5*torch.mean(kldyn).item(), xinit.detach().numpy(),xinit_ref.detach().numpy()
        # kl_static, mean0, var0 
    
    def update_ref_statistics(self,time=None,testing_samples=None,x0_samples=None):
        """
        Generate samples from the intermediate distribution by evolving the SDE
        
        ====
        Parameters
            time : float
                upper limit for integration
            
            testing_samples : int
                number of samples. Default to value in data.yaml
        Returns
            xinit : arr
                samples from the final distribution under the current drift
        ====
        """

        if time is None: 
            time = self.boundary.params["Tf"] 
            #t_steps = self.sde_params["tsteps"]
        #else: 
            #scale number of steps for the fraction of the time
        t_steps = int(np.floor(self.sde_params["tsteps"]*(time/self.boundary.params["Tf"])))
      
        if testing_samples == None: #default to the integrator 
            testing_samples = self.testing_samples

        self.net.eval()
        
        with torch.no_grad():

            t_vec = torch.linspace(0,time,t_steps)

            if x0_samples == None: #default to the integrator 
                xinit = torch.tensor(self.boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)
            else:
                xinit = x0_samples.detach().clone()

            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
                
                xinit,_ = self.integration_step(xinit,dt,t_vec[i])
         
        return xinit

    def save_cumulants(self,save_array,temp):
      """Append the computed cumulants to a new array"""
      
      return np.append(save_array,np.array([[np.nanmean(temp[...,0]),
                                              np.nanmean(temp[...,1]),
                                              np.nanvar(temp[...,0]),
                                              np.nanvar(temp[...,1]),
                                              np.correlate(temp[...,0],temp[...,1])[0]/(len(temp[...,0]))]]),axis=0)#cross corr

    def compute_sde(self,
                        final_time,reference_sol=None):
        """
        Compute the evolution of the SDE while saving cumulants 

        ====
        args:
          final_time: when to stop the integration for intermediate time plots
        ====
        returns:
          t_vec : ndarray
            times of evolution
          xinit.numpy() : ndarray
            point cloud of final positions under nn evolution
          xref.numpy() : ndarray
            point cloud of final positions under ref evolution
          cumulants_nn : ndarray
            cumulants computed during nn evolution
          cumulants_ref : ndarray
            cumulants computed during ref evolution
        """
            
        #pdb.set_trace()
        if reference_sol.duref_net is not None:
            t_vec = self.generate_time_grid()
        else:
            t_vec = reference_sol.get_underdamped_tgrid()

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
               
                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i])
        
                xref = self.reference_integration_step(xref,dt,t_vec[i],reference_sol)
                
                cumulants_nn = self.save_cumulants(cumulants_nn,xinit.numpy())
                cumulants_ref = self.save_cumulants(cumulants_ref,xref.numpy())
        
        return t_vec, xinit.numpy(), xref.numpy(), cumulants_nn, cumulants_ref


    def compute_pdf(self,xinit,Tf=None):
        """
        Computes an estimate for the final joint distribution using Girsanov theorem 
        Does not track gradients 
        Following 2411.08518 (see Section 3 for example). 

        ====
        Parameters
            xinit : array
                values of x0 on which to evaluate rho
            Tf : float 
                time for the plot
        Returns: 
            pdfout : arr
                estimate of the pdf at time Tf at locations xinit
        ====
        """

        num_samples = xinit.shape[0]

        #set default time to final time
        if Tf==None:
            Tf = self.boundary.params["Tf"]

        with torch.no_grad():

            t_vec = self.generate_time_grid(Tf,tsteps=self.sde_params["tsteps"])#tsteps=16)

            #preallocate girsanov integrals
            girs = torch.zeros((num_samples,1),requires_grad=False,dtype=torch.float32)

            for i in range(1,len(t_vec)):#self.sde_params["tsteps"]+1):
                                
                #get timestep
                dt = np.abs(t_vec[-i-1]-t_vec[-i])
                
                dw = self.generate_innovation(dt,num_samples)
            
                #evaluate neural network approximated drift 
                du_bar = self.net(torch.column_stack((t_vec[-i]*torch.ones((num_samples,1),dtype=torch.float32)
                                                           ,xinit[...,[0]])))
                
                #one backward step of euler maruyama scheme of stochastic evolution
                xinit = self.boundary.girsanov_scheme(xinit,du_bar,-dt,-dw,t_vec[-i])#-dt,-dw,t_vec[-i])
                
                #add step to stochastic integral
                girs = girs + (dw*xinit[...,[1]])/(2**0.5) + (dt * (xinit[...,[1]])**2)/4
                          
            pdfout = self.boundary.p_initial(xinit[...,[0]],xinit[...,[1]])*torch.exp(-girs)

        return pdfout
    
class VerletSDE(SDE):

    def __init__(self,
                    boundary_data,
                    terminal_cost=None, 
                    net=None,
                    sde_params=None,
                    device="cpu"):
        
        SDE.__init__(self,boundary_data,
                            terminal_cost, 
                            net,
                            sde_params,
                            device)

        self.testing_samples = self.sde_params["testsamples"]

    @torch.no_grad()
    def integration_step(self,xinit,dt,t):
        
        dw = self.generate_innovation(dt,self.testing_samples)

        #update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #compute du 
        du = self.net(torch.column_stack((t*torch.ones((self.testing_samples,1),requires_grad=False,dtype=torch.float32)
                                                            ,xinit[...,[0]]))).detach()
                    
        #Update the velocity
        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
        #position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)
            
        return xinit, du 
    
    @torch.no_grad()
    def reference_integration_step(self,xinit,dt,t,reference_sol=None):
        
        dw = self.generate_innovation(dt,self.testing_samples)

        #update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        if reference_sol.duref_net is not None:
            du = reference_sol.duref_net(torch.column_stack((t*torch.ones((self.testing_samples,1)),xinit[:,0,None])))
        else: 
            du = torch.as_tensor(reference_sol.get_ref_drift_gaussian(t),dtype=torch.float32)*xinit[...,[0]]
            #torch.zeros_like(xinit_ref[:,0,None],dtype=torch.float32,requires_grad=False)
                    
        #Update the velocity
        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
        #position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)
            
        return xinit 

    def verlet_half_step(self,xinit,du_bar, dt,dw,t=0):
              
        #Update the velocity
        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du_bar)*dt \
                                + self.boundary.sigma0*dw
        #position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*dt/2
        
        return xinit
    
    def loss_fn(self):
        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters

        ====
        """
        
        #self.net.train()
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)
        
        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        abar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        dw = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)

        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
                
            dw = self.generate_innovation(dt,self.sde_params["batchsamples"])
            
            #position half step 
            xinit[...,[0]] += self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)      
            
            #get neural network drift
            du = self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                                                        ,xinit[...,[0]])))
                           
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True)**2) 
            
            c = c + (du*dw)
            du_bar = du.detach() #copy without gradient 

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                    
                #integrate one step 
                xinit = self.verlet_half_step(xinit, du_bar, dt, dw, t_vec[i])

        with torch.no_grad():
            #compute terminal cost
            gvals = self.TerminalCost.gfun(xinit.clone().detach())
                
        #compute value of Lhat
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + gvals) * c))
        
        return Lhat
    

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

