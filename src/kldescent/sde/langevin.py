"""Implement the integration of SDE processes"""

import numpy as np
import torch

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
        Compute sample evolutions of the dynamics and save the metrics for
        recording the training process.

        Parameters
        ----------
        benchmark : object, optional
            Benchmark solution handler.

        Returns
        -------
        kldyn : float
            Dynamic KL divergence computed along trajectories.
        """

        with torch.no_grad():

            self.net.eval()

            t_vec = self.generate_time_grid(benchmark)
            test_samples_temp = 2048 #self.testing_samples
            #initialise x values as samples from assigned initial distribution
            xinit = self.boundary.generate_samples(test_samples_temp,"initial").to(self.device)
            xinit_ref = torch.clone(xinit).to(self.device)
            du_bar = self.net(torch.cat((torch.zeros_like(xinit[...,[0]],dtype=torch.float32).to(self.device)
                                                                        ,xinit[...,[0]]),dim=-1)).detach()
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
        Generate samples from the intermediate distribution by evolving the SDE.

        Parameters
        ----------
        time : float, optional
            Upper limit for integration.
        testing_samples : int, optional
            Number of samples. Default to value in data.yaml.

        Returns
        -------
        xinit : np.ndarray
            Samples from the final distribution under the current drift.
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
            du_bar = self.net(torch.cat((torch.zeros_like(xinit[...,[0]],dtype=torch.float32).to(self.device)
                                                   ,xinit[...,[0]]),dim=-1)).detach()

            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])

                xinit,du_bar = self.integration_step(xinit,dt,t_vec[i],du_bar)

        return xinit

    @torch.no_grad
    def compute_sde_cumulants(self,benchmark=None,mcsamples=None):
        """
        Compute the evolution of the SDE while saving cumulants.

        Parameters
        ----------
        benchmark : object, optional
            Benchmark solution handler; used to compute reference samples.
        mcsamples : int, optional
            Number of Monte Carlo samples.

        Returns
        -------
        t_vec : np.ndarray
            Times of evolution.
        xinit : np.ndarray
            Point cloud of final positions under NN evolution.
        xref : np.ndarray
            Point cloud of final positions under reference evolution.
        cumulants_nn : np.ndarray
            Cumulants computed during NN evolution.
        cumulants_ref : np.ndarray
            Cumulants computed during reference evolution.
        """
            
        self.net.eval()

        if mcsamples is None: #default to test samples
            mcsamples = self.sde_params["testsamples"]

        t_vec = self.generate_time_grid(benchmark)
        
        #initialise x values at \delta_0 (i.e. x=0)
        xinit = self.boundary.generate_samples(mcsamples,"initial").to(self.device)
        xref = torch.clone(xinit).to(self.device)
        du_bar = self.net(torch.cat((torch.zeros_like(xinit[...,[0]],dtype=torch.float32).to(self.device)
                                                                ,xinit[...,[0]]),dim=-1)).detach()
                                   
        
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
                #try:
                #    du = torch.as_tensor(benchmark.du_benchmark(t,xinit[...,[0]]),dtype=torch.float32)
                #except: 
                du = torch.zeros_like(xinit[...,[0]],dtype=torch.float32,requires_grad=False)
          
        return torch.zeros_like(xinit[...,[0]],dtype=torch.float32,requires_grad=False)
        
class EulerMaruyamaSDE(SDEBase):

    def __init__(self,
                    boundary_data,
                    net=None,
                    sde_params=None):

        SDEBase.__init__(self,boundary_data=boundary_data,sde_params=sde_params)

        self.net = net
        self.testing_samples =self.sde_params["testsamples"]
        self.device = self.net.device

    @torch.no_grad()
    def girsanov_step(self, xinit, du_bar, dt, dw):
        """Euler-Maruyama step under the reference measure (no damping), used in backward Girsanov integration."""
        xinit[...,[0]] = xinit[...,[0]] - self.boundary.params["epsilon"] * xinit[...,[1]] * dt
        xinit[...,[1]] = xinit[...,[1]] - self.boundary.sigma0 * du_bar * dt - self.boundary.sigma0 * dw
        return xinit

    @torch.no_grad()
    def loss_intermediate_step(self,xinit,dt,t,du_prev):

        return xinit

    @torch.no_grad()
    def generate_innovation(self,dt,size_like):
        """generate a step of brownian motion"""
        return torch.sqrt(dt)*torch.randn_like(size_like,requires_grad=False,dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def loss_final_step(self,xinit,dt):

        return xinit

    def loss_integration_step(self,xinit,dt,t,du_prev,dw):

        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt)

        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du_prev)*dt \
                                                + self.boundary.sigma0*dw

        return xinit

    def integration_step(self,xinit,dt,t,du_prev):
        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])

        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt)

        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du_prev)*dt \
                                                + self.boundary.sigma0*dw

        #compute next du
        du = self.net(torch.cat(((t+dt)*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]),dim=-1)).detach()
        return xinit, du

    @torch.no_grad()
    def benchmark_integration_step(self,xinit,dt,t,benchmark):
        
        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])

        du = self.compute_benchmark_drift(xinit,t,benchmark)
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt)

        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du)*dt \
                                                + self.boundary.sigma0*dw
            
        return xinit

class BAOAB(SDEBase):

    def __init__(self,
                    boundary_data,
                    net=None,
                    sde_params=None):

        SDEBase.__init__(self,boundary_data=boundary_data,sde_params=sde_params)

        self.net = net
        self.testing_samples =self.sde_params["testsamples"]
        self.device = self.net.device


    @torch.no_grad()
    def loss_integration_step(self,xinit,dt,t,du_prev,dw):

        #Update the velocity
        #B: deterministic kick
        xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du_prev)*(dt/2)
        
        #A: update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #O : solve OU step
        xinit[...,[1]] = torch.exp(-dt)*xinit[...,[1]] + self.boundary.sigma0*dw
        
        #A: position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #compute du 
        #du = self.net(torch.cat(((t+dt)*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
        #                                                    ,xinit[...,[0]]),dim=-1)).detach()
        
        #B: deterministic kick
        #xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du)*(dt/2)
        
        return xinit

    @torch.no_grad()
    def generate_innovation(self,dt,size_like):
        """generate a step of brownian motion"""
        return torch.sqrt((1-torch.exp(-2*dt))/2)*torch.randn_like(size_like,requires_grad=False,dtype=torch.float32).to(self.device)

    
    @torch.no_grad()
    def loss_intermediate_step(self,xinit,dt,t,du_prev):

        #compute du 
        #du = self.net(torch.cat(((t+dt)*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
        #                                                    ,xinit[...,[0]]),dim=-1)).detach()
        
        if t>0:
            #B: final deterministic kick
            xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du_prev)*(dt/2)
            
        return xinit
   

    @torch.no_grad()
    def loss_final_step(self,xinit,dt):

        #compute du 
        du = self.net(torch.cat((self.boundary.params["Tf"]*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]),dim=-1)).detach()
        xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du)*(dt/2)
        
        return xinit
   

    @torch.no_grad()
    def integration_step(self,xinit,dt,t,du_prev):
        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])

        #Update the velocity
        #B: deterministic kick
        xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du_prev)*(dt/2)
        
        #A: update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #O : solve OU step
        xinit[...,[1]] = torch.exp(-dt)*xinit[...,[1]] + self.boundary.sigma0*dw
        
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

    @torch.no_grad()
    def girsanov_step(self, xinit, du_bar, dt, dw):
        """BAOAB Girsanov step: B-A-O-A-B splitting under the reference measure."""
        #B: drift kick
        xinit[...,[1]] = xinit[...,[1]] - self.boundary.sigma0 * du_bar * (dt/2)
        #A: position half-step
        xinit[...,[0]] = xinit[...,[0]] - self.boundary.params["epsilon"] * xinit[...,[1]] * (dt/2)
        #O: OU noise step
        xinit[...,[1]] = -torch.exp(-dt)*xinit[...,[1]] - self.boundary.sigma0*dw
        #A: position half-step
        xinit[...,[0]] = xinit[...,[0]] - self.boundary.params["epsilon"] * xinit[...,[1]] * (dt/2)
        #B: drift kick
        xinit[...,[1]] = xinit[...,[1]] - self.boundary.sigma0 * du_bar * (dt/2)
        return xinit

    def verlet_half_step(self,xinit,du_bar,dt,dw,t=0):

        #Update the velocity
        #B: deterministic kick
        xinit[...,[1]] = xinit[...,[1]] + (self.boundary.sigma0*du_bar)*(dt/2)

        #A: update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #O : solve OU step
        xinit[...,[1]] = torch.exp(-dt)*xinit[...,[1]] + self.boundary.sigma0*dw

        #A: position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #final B step is part of main loop

        return xinit

class VerletSDE(SDEBase):

    def __init__(self,
                    boundary_data,
                    net=None,
                    sde_params=None):

        SDEBase.__init__(self,boundary_data)

        self.net = net
        self.testing_samples =self.sde_params["testsamples"]
        self.device = self.net.device


    @torch.no_grad()
    def loss_intermediate_step(self,xinit,*_):

        return xinit

    @torch.no_grad()
    def loss_final_step(self,xinit,_):

        return xinit

    @torch.no_grad()
    def loss_integration_step(self,xinit,dt,t,du_prev,dw):

        #update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #Update the velocity
        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du_prev)*dt \
                                                + self.boundary.sigma0*dw
        #position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        return xinit

    @torch.no_grad()
    def generate_innovation(self,dt,size_like):
        """generate a step of brownian motion"""
        return torch.sqrt(dt)*torch.randn_like(size_like,requires_grad=False,dtype=torch.float32).to(self.device)

    
    @torch.no_grad()
    def integration_step(self,xinit,dt,t,du_prev):
        dw = self.generate_innovation(dt,size_like=xinit[...,[0]])

        #update positions by half step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #Update the velocity
        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du_prev)*dt \
                                                + self.boundary.sigma0*dw
        #position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*(dt/2)

        #compute next du
        du = self.net(torch.cat(((t+dt)*torch.ones_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(self.device)
                                                            ,xinit[...,[0]]),dim=-1)).detach()

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

    @torch.no_grad()
    def girsanov_step(self, xinit, du_bar, dt, dw):
        """Verlet Girsanov step: position half-steps around a friction-free momentum update."""
        #A: position half-step
        xinit[...,[0]] = xinit[...,[0]] - self.boundary.params["epsilon"] * xinit[...,[1]] * (dt/2)
        #momentum update (no friction)
        xinit[...,[1]] = xinit[...,[1]] - self.boundary.sigma0 * du_bar * dt - self.boundary.sigma0 * dw
        #A: position half-step
        xinit[...,[0]] = xinit[...,[0]] - self.boundary.params["epsilon"] * xinit[...,[1]] * (dt/2)
        return xinit

    def verlet_half_step(self,xinit,du_bar, dt,dw,t=0):

        #Update the velocity
        xinit[...,[1]] = xinit[...,[1]] - (xinit[...,[1]] - self.boundary.sigma0*du_bar)*dt \
                                + self.boundary.sigma0*dw
        #position half-step
        xinit[...,[0]] = xinit[...,[0]] + self.boundary.params["epsilon"]*xinit[...,[1]]*dt/2
        
        return xinit
        

if __name__=="__main__":
    from kldescent.model import nn
    from kldescent.sde.boundary import UnderdampedHalfGaussian
    from kldescent.utils.parser import integrator_params

    boundaryconditions = UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    integrator = EulerMaruyamaSDE(boundaryconditions)

    integrator.update_ref_statistics(10,debug=True)

