#this file computes the dynamics 
import numpy as np 
import torch
import sde.kl as kl
import pdb

class SDEOverdamped():
    def __init__(self,batch_samples,epsilon,Tf,t_steps,boundary_data,logger):
        self.batch_samples = batch_samples # number of samples in each step 
        self.epsilon = epsilon 
        self.Tf = Tf #final time
        self.t_steps = t_steps #number of timesteps in discretisation
        self.boundary = boundary_data
        self.logger = logger

    def generate_time_grid(self):

        #samples the interval [0,Tf] and returns sorted set of values to be used in integration

        return np.concatenate([[0],
                                np.sort(np.random.uniform(0, self.Tf, self.t_steps-1)),
                                [self.Tf]])


    def g(self,q): 

        return self.boundary.g(q)#torch.tensor(self.boundary.g(q),requires_grad=False,dtype=torch.float32)

    def euler_maruyama_step_refsolution(self,xn,currtime,dt,dw):
        #takes a step of the euler maruyama using the reference solution 
      
        ducurr = self.boundary.du_ref(xn,currtime)
        #takes one step forward in the euler maruyama discretisation
        return xn + (self.boundary.b(xn,t) \
                    + self.boundary.sigma0*ducurr)*dt + self.boundary.sigma0*dw

    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):
        #takes one step forward in the euler maruyama discretisation
        
        return xn + (self.boundary.b(xn,t) \
                     + self.boundary.sigma0*du_bar)*dt + self.boundary.sigma0*dw


    def testing_time_grid(self):
        #get an equally spaced time grid which can be used with the reference solution
        
        ##TODO Update so this can also be used with reference solution of the Halfbridge
        return np.linspace(0,self.Tf,self.t_steps)

    def generate_innovation(self,dt):
        return np.sqrt(dt)*torch.randn((self.batch_samples,1),requires_grad=False,dtype=torch.float32)

    def simulation_free(self,net,logflag,epoch):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network
        ====

        """
        
        #add as command line option??
        testing_mode = False
        
        if testing_mode:
            t_vec = self.testing_time_grid()
        else:
            t_vec = self.generate_time_grid()

        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.tensor(self.boundary.generate_samples_q((self.batch_samples,1),"initial"),requires_grad=False,dtype=torch.float32)

        #set up a,abar,c,cbar
        a = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        abar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        
        #self.generate_time_grid()
        for i in range(0,self.t_steps-2):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt)
            
            if testing_mode:
                du = torch.tensor(self.boundary.du_ref(xinit,t_vec[i]).reshape((self.batch_samples,1)),
                                    requires_grad=False,
                                    dtype=torch.float32)
            else:
                du = net.forward(torch.column_stack((t_vec[i]*torch.ones(self.batch_samples,dtype=torch.float32)
                                                    ,xinit)))
                    
            a = a + dt*(torch.linalg.norm(du, axis=1) ** 2) 
            
            c = c + torch.matmul(torch.unsqueeze(du, 1),
                                          torch.unsqueeze(dw, 2),
                                                ).reshape((self.batch_samples,1)) #du*dw
            du_bar = du.detach()#.numpy() #untracked copy

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                
            xinit = self.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])


        #compute value of Lhat
        #see equation (10) and (11) in ref. 2410.05163
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + self.g(xinit)) * c))

        with torch.no_grad():    
            if logflag:
                self.logger.save_data(Lhat,xinit.detach().cpu().numpy(),
                                    net,epoch)

        return Lhat


if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
