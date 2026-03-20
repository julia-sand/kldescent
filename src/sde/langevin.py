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
    def uniform_random_time_grid(self,Tf,tsteps):
        """
        Generate a randomly sampled time grid to be used in integration
        """
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
      temp = torch.cat(rows, dim=0).cpu().numpy()
      print(temp.shape)
      return temp

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

            t_vec = self.generate_time_grid(benchmark)
            #initialise x values as samples from assigned initial distribution
            xinit = torch.tensor(self.boundary.generate_samples(self.testing_samples,"initial"),requires_grad=False,dtype=torch.float32).to(self.device)
            xinit_ref = torch.clone(xinit).to(self.device)

            #preallocate storing kl divergence 
            kldyn = torch.zeros((self.testing_samples,1),requires_grad=False,dtype=torch.float32).to(self.device)
            
            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
            
                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i])
        
                xinit_ref = self.benchmark_integration_step(xinit_ref,dt,t_vec[i],benchmark)
                
                kldyn = kldyn + dt*(du_bar**2)
            
        return 0.5*torch.mean(kldyn).item(), xinit.detach().cpu().numpy(),xinit_ref.detach().cpu().numpy()
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

            t_vec = self.uniform_random_time_grid(time,t_steps)#torch.linspace(0,time,t_steps)
            

            if x0_samples == None: #default to the integrator 
                xinit = torch.tensor(self.boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32).to(self.device)
            else:
                xinit = x0_samples.clone().to(self.device)

            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
                
                xinit,_ = self.integration_step(xinit,dt,t_vec[i])
         
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
            
        if mcsamples is None: #default to test samples
            mcsamples = self.sde_params["testsamples"]

        t_vec = self.generate_time_grid(benchmark)
        
        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.tensor(self.boundary.generate_samples(mcsamples,"initial"),requires_grad=False,dtype=torch.float32).to(self.device)
        xref = torch.clone(xinit).to(self.device)

        #save the cumulants of the evolutions
        cumulants_nn = []#self.save_cumulants(torch.empty((0,5),requires_grad=False,dtype=torch.float32),xinit).to(self.device)
        cumulants_ref = []#self.save_cumulants(torch.empty((0,5),requires_grad=False,dtype=torch.float32),xref).to(self.device)
        cumulants_nn.append(self.compute_cumulant_row(xinit))
        cumulants_ref.append(self.compute_cumulant_row(xinit))

        self.net.eval()

        with torch.no_grad():

            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
               
                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i])
        
                xref = self.benchmark_integration_step(xref,dt,t_vec[i],benchmark)
                
                cumulants_nn.append(self.compute_cumulant_row(xinit))#self.save_cumulants(cumulants_nn,xinit)
                cumulants_ref.append(self.compute_cumulant_row(xref))
        
        #save as numpy arrays and return to cpu for plots
        cumulants_nn = self.save_cumulants(cumulants_nn)
        cumulants_ref = self.save_cumulants(cumulants_ref)

        return t_vec, xinit.cpu().numpy(), xref.cpu().numpy(), cumulants_nn,cumulants_ref #cumulants_nn.cpu(), cumulants_ref.cpu()

class EulerMaruyamaSDE(SDEBase):

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
        self.device = self.net.device

    @torch.no_grad
    def compute_benchmark_drift(self,xinit,t,benchmark):
        
        if benchmark is None:
            du = torch.zeros_like(xinit[...,[0]],dtype=torch.float32,requires_grad=False)
        else:
            if benchmark.du_net_benchmark is not None:
                du = benchmark.du_net_benchmark(torch.column_stack((t*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]])))
            else:
                du = torch.as_tensor(benchmark.get_drift_gaussian(t),dtype=torch.float32)*xinit[...,[0]]
            
        
        return du

    def integration_step(self,xinit,dt,t):

        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])#self.testing_samples)

        #compute du 
        du = self.net(torch.column_stack((t*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]))).detach()
                    
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
        t_vec = self.uniform_random_time_grid(self.boundary.params["Tf"],self.sde_params["tsteps"])

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32).to(self.device)
        
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
            du = self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
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
            gvals = self.TerminalCost.gfun(xinit.detach().clone(),self.net,self.sde_params["testsamples"])

        #compute value of Lhat
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + gvals) * c))
        
        return Lhat
    
class VerletSDE(SDEBase):

    def __init__(self,
                    boundary_data,
                    terminal_cost=None, 
                    net=None,
                    sde_params=None,
                    device="cpu"):
        
        SDEBase.__init__(self,boundary_data)

        self.TerminalCost = terminal_cost
        self.net = net 
        self.testing_samples =self.sde_params["testsamples"]
        
    @torch.no_grad()
    def integration_step(self,xinit,dt,t):

        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])

        #update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #compute du 
        du = self.net(torch.column_stack((t*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]))).detach()
                    
        #Update the velocity
        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
        #position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)
            
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
        t_vec = self.uniform_random_time_grid(self.boundary.params["Tf"],self.sde_params["tsteps"])

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32).to(self.device)
        
        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        abar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        dw = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)

        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
                
            dw = self.generate_innovation(dt,size_like=xinit[...,[0]])
            
            #position half step 
            xinit[...,[0]] += self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)      
            
            #get neural network drift
            du = self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32).to(self.device)
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
    from src.utils.parser import integrator_params

    boundaryconditions = UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    integrator = EulerMaruyamaSDE(boundaryconditions)

    integrator.update_ref_statistics(10,debug=True)

