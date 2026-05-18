"""Implement the integration of SDE processes"""

import numpy as np 
import torch

import kldescent.sde.kl as kl
from kldescent.utils.parser import integrator_params

class SDEBase():
    """
    Base class for integrating the sde forward and computing the loss function 
    """

    def __init__(self,boundary_data,
                    sde_params=None):
                    
        self.boundary = boundary_data

        #get params 
        if sde_params is not None:
            self.sde_params = sde_params #take params from given yaml
        else:
            self.sde_params = integrator_params() #parse command line
    
    @torch.no_grad
    def generate_time_grid(self,benchmark=None,Tf=None,tsteps=None):
        """Generate time grid for comparison with benchmark"""

        if Tf ==None:
            Tf = self.boundary.params["Tf"]
        if tsteps==None:
            tsteps= self.sde_params["tsteps"]
            
        if benchmark is None:
            return self.uniform_random_time_grid(Tf,tsteps) #use random time grid
        else: 
            if benchmark.du_net_benchmark is not None:
                return self.uniform_random_time_grid(Tf,tsteps) #use random time grid
            else: 
                #use fixed times from csv
                return torch.from_numpy(benchmark.get_timegrid()).to(self.device)
        
    @torch.no_grad
    def uniform_random_time_grid(self,Tf=None,tsteps=None):
        """
        Generate a randomly sampled time grid to be used in integration
        """
        if Tf is None:
            Tf = self.boundary.params["Tf"]
        if tsteps is None:
            tsteps = self.sde_params["tsteps"]

        return torch.concatenate([torch.zeros(1),
                                Tf*torch.sort(torch.rand(tsteps,dtype=torch.float32,requires_grad=False))[0],
                                Tf*torch.ones(1)]).to(self.device)   
    
    @torch.no_grad()
    def generate_innovation(self,dt,size_like):
        """generate a step of brownian motion"""
        return torch.sqrt(dt)*torch.randn_like(size_like,requires_grad=False,dtype=torch.float32).to(self.device)

    @torch.no_grad 
    def cross_corr(self,temp):
        return torch.sum((temp[...,0]-torch.nanmean(temp[...,0]))*(temp[...,1]-torch.nanmean(temp[...,1])))/len(temp[...,0])

    @torch.no_grad
    def compute_cumulant_row(self,temp):
        return torch.tensor([[torch.nanmean(temp[...,0]),
                                              torch.nanmean(temp[...,1]),
                                              torch.var(temp[...,0]),
                                              torch.var(temp[...,1]),
                                              self.cross_corr(temp)]])

    @torch.no_grad
    def save_cumulants(self,rows):
      """Append the computed cumulants to a new array and return in numpy"""
      return torch.cat(rows, dim=0).cpu().numpy()

    @torch.no_grad
    def compute_metrics(self,benchmark=None):
        """
        Computes sample evolutions of the dynamics and saves the metrics for 
        recording the training process

        ====
        Parameters:
            benchmark : class
                benchmark 
        Returns:
            kldyn : float
                Dynamic KL divergence computed along trajectories. 
        ====
        """

        with torch.no_grad():

            self.net.eval()

            t_vec = self.generate_time_grid(benchmark)
            test_samples_temp = 2048 #self.testing_samples
            #initialise x values as samples from assigned initial distribution
            xinit = self.boundary.generate_samples(test_samples_temp,"initial").to(self.device)
            xinit_ref = torch.clone(xinit).to(self.device)

            du_bar = 
            
            #preallocate storing kl divergence 
            kldyn = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
            
            for i in range(0,len(t_vec)-1):
                #get time interval
                dt = (t_vec[i+1]-t_vec[i])
            
                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i],du_bar)
        
                xinit_ref = self.benchmark_integration_step(xinit_ref,dt,t_vec[i],benchmark)
                
                #add to integral
                kldyn = kldyn + dt*(du_bar**2)
            
        return 0.5*torch.mean(kldyn).detach().cpu().numpy(), xinit.cpu().numpy(),xinit_ref.cpu().numpy()
        # kl_static, mean0, var0 
    
    @torch.no_grad
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

        self.net.eval()
        
        if time is None: 
            time = self.boundary.params["Tf"] 
            #t_steps = self.sde_params["tsteps"]
        #else: 
            #scale number of steps for the fraction of the time
        t_steps = int(np.floor(self.sde_params["tsteps"]*(time/self.boundary.params["Tf"])))
      
        if testing_samples == None: #default to the integrator 
            testing_samples = self.testing_samples

        with torch.no_grad():

            t_vec = self.uniform_random_time_grid(time,t_steps)#torch.linspace(0,time,t_steps)
            

            if x0_samples == None: #default to the integrator 
                xinit = self.boundary.generate_samples(testing_samples,"initial").to(self.device)
            else:
                xinit = x0_samples.clone().to(self.device)

            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
                
                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i],du_bar)
         
        return xinit

    @torch.no_grad
    def compute_sde_cumulants(self,benchmark=None,mcsamples=None):
        """
        Compute the evolution of the SDE while saving cumulants 

        ====
        args:
            referencesol: class
                compute reference samples
            mcsamples : int
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
            
        self.net.eval()

        if mcsamples is None: #default to test samples
            mcsamples = self.sde_params["testsamples"]

        t_vec = self.generate_time_grid(benchmark)
        
        #initialise x values at \delta_0 (i.e. x=0)
        xinit = self.boundary.generate_samples(mcsamples,"initial").to(self.device)
        xref = torch.clone(xinit).to(self.device)

        #save the cumulants of the evolutions
        cumulants_nn = []#self.save_cumulants(torch.empty((0,5),requires_grad=False,dtype=torch.float32),xinit).to(self.device)
        cumulants_ref = []#self.save_cumulants(torch.empty((0,5),requires_grad=False,dtype=torch.float32),xref).to(self.device)
        cumulants_nn.append(self.compute_cumulant_row(xinit))
        cumulants_ref.append(self.compute_cumulant_row(xref))

        with torch.no_grad():

            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
               
                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i],du_bar)
        
                xref = self.benchmark_integration_step(xref,dt,t_vec[i],benchmark)
                
                cumulants_nn.append(self.compute_cumulant_row(xinit))
                cumulants_ref.append(self.compute_cumulant_row(xref))
        
        #save as numpy arrays and return to cpu for plots
        cumulants_nn = self.save_cumulants(cumulants_nn)
        cumulants_ref = self.save_cumulants(cumulants_ref)

        return t_vec.cpu().numpy(), xinit.cpu().numpy(), xref.cpu().numpy(), cumulants_nn,cumulants_ref #cumulants_nn.cpu(), cumulants_ref.cpu()

    @torch.no_grad
    def compute_benchmark_drift(self,xinit,t,benchmark):
        
        if benchmark is None:
            du = torch.zeros_like(xinit[...,[0]],dtype=torch.float32,requires_grad=False)
        else:
            if benchmark.du_net_benchmark is not None:
                du = benchmark.du_net_benchmark(torch.cat((t*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]),dim=-1))
            else:
                try:
                    du = torch.as_tensor(benchmark.get_drift_gaussian(t),dtype=torch.float32)*xinit[...,[0]]
                except: 
                    du = torch.zeros_like(xinit[...,[0]],dtype=torch.float32,requires_grad=False)
          
        return du
        
class EulerMaruyamaSDE(SDEBase):

    def __init__(self,
                    boundary_data,
                    terminal_cost=None, 
                    net=None,
                    sde_params=None):
        
        SDEBase.__init__(self,boundary_data=boundary_data,sde_params=sde_params)

        self.TerminalCost = terminal_cost
        self.net = net 
        self.testing_samples =self.sde_params["testsamples"]
        self.device = self.net.device


    def integration_step(self,xinit,dt,t,_):

        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])#self.testing_samples)

        #compute du 
        du = self.net(torch.cat((t*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]),dim=-1)).detach()
                    
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt)

        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
            
        return xinit, du 

    @torch.no_grad()
    def benchmark_integration_step(self,xinit,dt,t,benchmark):
        
        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])

        du = self.compute_benchmark_drift(xinit,t,benchmark)
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt)

        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
            
        return xinit

    def loss_fn(self):
        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters

        ====
        """
        
        self.net.train()
        
        t_vec = self.uniform_random_time_grid()
        
        #initialise x values 
        xinit = self.boundary.generate_samples(self.sde_params["batchsamples"],"initial").to(self.device)

        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
        abar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32).to(self.device)
        bbar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32).to(self.device)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
        dw = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32).to(self.device)

        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,xinit[...,[0]])

            #get neural network drift
            du = self.net(torch.cat((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
                                                    ,xinit[...,[0]]),dim=-1))

            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True) ** 2) 
            
            c = c + (du*dw)

            du_bar = du.detach() #copy without gradient 

            with torch.no_grad(): #no gradient 

                #abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                #abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                #bbar = bbar #+ dt*(self.boundary.f(xinit,t_vec[i])) 
                
                xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])


        with torch.no_grad():
            #compute terminal cost
            gvals = self.TerminalCost.gfun(xinit.detach().clone(),self.net,self.sde_params["testsamples"]).to(self.device)

            a_bar = a.detach().clone()

        #compute value of Lhat
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + gvals) * c))
        
        return Lhat
    
class VerletSDE(SDEBase):

    def __init__(self,
                    boundary_data,
                    terminal_cost=None, 
                    net=None,
                    sde_params=None):
        
        SDEBase.__init__(self,boundary_data)

        self.TerminalCost = terminal_cost
        self.net = net 
        self.testing_samples =self.sde_params["testsamples"]
        self.device = self.net.device

    @torch.no_grad()
    def integration_step(self,xinit,dt,t,du_prev):

        #sample dw
        dw = self.generate_innovation(1,size_like=xinit[...,[0]])

        #Update the velocity
        #B: deterministic kick
        xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du_prev)*(dt/2)
        
        #A: update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #O : solve OU step
        xinit[...,[1]] = torch.exp(-dt)*xinit[...,[1]] + self.boundary.sigma0*torch.sqrt(1-torch.exp(-2*dt))*dw
        
        #A: position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #compute du 
        du = self.net(torch.cat(((t+dt)*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]),dim=-1)).detach()
        
        #B: deterministic kick
        xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du)*(dt/2)
        
        return xinit, du 
    
    @torch.no_grad()
    def benchmark_integration_step(self,xinit,dt,t,benchmark=None):
        
        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])

        #update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)
        
        du = self.compute_benchmark_drift(xinit,t,benchmark)
        
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
        
        self.net.train()
        t_vec = self.uniform_random_time_grid()

        #initialise x values 
        xinit = self.boundary.generate_samples(self.sde_params["batchsamples"],"initial").to(self.device)
        
        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
        abar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32).to(self.device)
        bbar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32).to(self.device)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
        dw = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32).to(self.device)

        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
                
            dw = self.generate_innovation(dt,size_like=xinit[...,[0]])
            
            #get neural network drift
            du = self.net(torch.cat((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
                                                        ,xinit[...,[0]]),dim=-1))
                           
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True)**2) 
            
            c = c + (du*dw)
            du_bar = du.detach() #copy without gradient 

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                #bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                    
                #integrate one step 
                xinit,_ = self.integration_step(xinit,dt,t,du_bar)
                #self.verlet_half_step(xinit, du_bar, dt, dw, t_vec[i])

        with torch.no_grad():
            #compute terminal cost
            gvals = self.TerminalCost.gfun(xinit.detach().clone(),self.net,self.sde_params["testsamples"]).to(self.device)
                
        #compute value of Lhat
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + gvals) * c))
        
        return Lhat
            

if __name__=="__main__":
    from kldescent.model import nn
    from kldescent.sde.boundary import UnderdampedHalfGaussian
    from kldescent.utils.parser import integrator_params

    boundaryconditions = UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    integrator = EulerMaruyamaSDE(boundaryconditions)

    integrator.update_ref_statistics(10,debug=True)

