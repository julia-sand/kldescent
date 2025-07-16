import numpy as np 
import torch

import src.sde.kl as kl
from ..utils.parser import integrator_params
#from src.utils.parser import integrator_params


class SDEOverdamped():
    def __init__(self,
                    boundary_data,
                    sde_params=None,
                    logger=None,
                    device="cpu"):
        self.boundary = boundary_data
        self.logger = logger

        #get params 
        if sde_params is not None:
            self.sde_params = sde_params #take params from given yaml
        else:
            self.sde_params = integrator_params() #parse command line

        if self.logger is not None:
            self.logger.log_dict(self.sde_params) #add to log
        
    def generate_time_grid(self):

        #samples the interval [0,Tf] and returns sorted set of values to be used in integration

        return np.concatenate([[0],
                                np.sort(np.random.uniform(0, self.boundary.params["Tf"], self.sde_params["tsteps"]-1)),
                                [self.boundary.params["Tf"]]])


    def testing_time_grid(self,final_time=None):

        if final_time==None:
            final_time = self.boundary.params["Tf"]
        
        #get an equally spaced time grid which can be used with the reference solution
        time_grid = self.boundary.get_reference_timegrid()
        if time_grid is not None:
            return time_grid
        else:
            return np.linspace(0,final_time,self.sde_params["tsteps"])

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
        
        net.train()
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)

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
           
            du = net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                                                    ,xinit[...,[0]])))
                    
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True) ** 2) 
            
            c = c + (du*dw)
            du_bar = du.detach()#.numpy() #untracked copy

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                
            xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])

        #compute value of Lhat
        #see equation (10) and (11) in ref. 2410.05163
        #Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + self.boundary.gfun(xinit)) * c))
        #Lhat = torch.mean(0.5 * a + (((1/4) * abar + bbar + self.boundary.gfun(xinit)) * c))
        Lhat = torch.mean(0.5 * a + ((abar + bbar + self.boundary.gfun(xinit)) * c))
        
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
    def compute_metrics(self,net,epoch,lout,lr):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network
        ====

        """

        net.eval()
        t_vec = self.testing_time_grid()
        testing_samples = self.sde_params["testsamples"] #4000 #reduced number of samples in computing the loss metrics

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

                du_bar = net(torch.column_stack((t_vec[i]*torch.ones((testing_samples,1),requires_grad=False,dtype=torch.float32)
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
        
        #update the reference statistics (ie mean and var of final distribution)
        #self.boundary.set_ref_statistics(np.nanmean(xinit[...,0]), np.nanvar(xinit[...,0]))

        if self.boundary.__class__.__name__== "UnderdampedHalfGaussian":
            #self.boundary.set_ref_statistics(torch.nanmean(xinit[...,0]).item(), torch.var(xinit[...,0]).item())
            pass
        if self.logger is not None:
            self.logger.log_epoch_data(net,
                                        epoch,
                                        lout,
                                        torch.mean(kldyn).item(),
                                        kl_static,
                                        lr,self.boundary.mean0,self.boundary.var0)
        
        #free up the tensors
        del xinit 
        del xinit_ref
        del du_ref
        del kldyn


        return 

    
    def update_ref_statistics(self,net,testing_samples=None,debug=False):


        """
        computes estimates of the metrics to update the g function 
        This should only run for the Half Bridge classes. 

        ====
        Parameters
        - net: neural network
        ====

        """
        if debug:
            pdb.set_trace()
        if self.boundary.__class__.__name__== "UnderdampedHalfGaussian":
            if testing_samples == None: #default to the integrator 
                testing_samples = self.sde_params["testsamples"] 

            net.eval()
            
            with torch.no_grad():

                t_vec = torch.linspace(0,self.boundary.params["Tf"],self.sde_params["tsteps"])#self.testing_time_grid()

                xinit = torch.tensor(self.boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)

                for i in range(0,len(t_vec)-1):
                    dt = (t_vec[i+1]-t_vec[i])
                    
                    dw = self.generate_innovation(dt,testing_samples)
                    #xdata = torch.column_stack(())
                    du_bar = net(torch.column_stack((t_vec[i]*torch.ones((testing_samples,1),requires_grad=False,dtype=torch.float32)
                                                            ,xinit[...,[0]])))
                    
                    xinit = self.boundary.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])
            

                self.boundary.set_ref_statistics(torch.nanmean(xinit[...,0]).item(), torch.var(xinit[...,0]).item())
            
            #free up the tensors
            del xinit 
            del du_bar

        else:
            pass

        return


    def get_net_grad(self,net,t):
        dx = 0.01
        grad_temp = (net(torch.column_stack((t*torch.ones(1),dt*torch.ones(1)))) - net(torch.column_stack((t*torch.ones(1),0*torch.ones(1)))))

        return grad_temp/dx

    def compute_sde(self,net,
                        plotter,
                        marginal_plot_time):

        """
        Computes the SDE, saves cumulants and plots 

        ====
        Parameters
        - net: neural network
        ====

        """
            
        #pdb.set_trace()
        t_vec = self.boundary.get_reference_timegrid(marginal_plot_time)

        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["testsamples"],"initial"),requires_grad=False,dtype=torch.float32)
        xref = torch.clone(xinit)

        #save the cumulants of the evolutions
        cumulants_nn = plotter.save_cumulants(np.empty((0,5)),xinit.numpy())
        cumulants_ref = plotter.save_cumulants(np.empty((0,5)),xref.numpy())

        with torch.no_grad():

            #self.generate_time_grid()
            for i in range(0,len(t_vec)-1):
                dt = (t_vec[i+1]-t_vec[i])
                
                dw = self.generate_innovation(dt,self.sde_params["testsamples"])
                du = net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["testsamples"],1),dtype=torch.float32)
                                                        ,xinit[...,[0]])))
                
                xinit = self.boundary.euler_maruyama_step(xinit,du,dt,dw,t_vec[i])
                xref = self.boundary.euler_maruyama_step(xref,self.boundary.du_ref(xref[:,[0]],t_vec[i]),dt,dw,t_vec[i])
                cumulants_nn = plotter.save_cumulants(cumulants_nn,xinit.numpy())
                cumulants_ref = plotter.save_cumulants(cumulants_ref,xref.numpy())
                
        plotter.marginal_distribution([xinit.numpy(),xref.numpy()],marginal_plot_time)
        plotter.plot_cumulants(t_vec,cumulants_nn,cumulants_ref,self.sde_params["testsamples"])
        
        #free up the tensors
        del xinit 
        del du

        return 

if __name__=="__main__":
    from src.model import nn
    from src.sde.boundary import UnderdampedHalfGaussian
    import pdb
    from src.utils.parser import integrator_params

    boundaryconditions = UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    integrator = SDEOverdamped(boundaryconditions)

    integrator.update_ref_statistics(net,10,debug=True)

