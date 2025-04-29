#this file computes the dynamics 
import numpy as np 
import numpy.random as npr
import torch
import sde.kl as kl
import pdb

class SDEOverdamped():
    def __init__(self,batch_samples,epsilon,Tf,t_steps,boundary_data):
        self.batch_samples = batch_samples # number of samples in each step 
        self.epsilon = epsilon 
        self.Tf = Tf #final time
        self.t_steps = t_steps #number of timesteps in discretisation
        self.boundary = boundary_data

    def generate_time_grid(self):

        #samples the interval [0,Tf] and returns sorted set of values to be used in integration

        return np.concatenate([[0],
                                np.sort(np.random.uniform(0, self.Tf, self.t_steps-1)),
                                [self.Tf]])


    def g(self,q): 
        #this is the penalty for the final time

        #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

        return (-torch.log(self.boundary.q_marginal_final(q)) - (q**2)/2)

    def euler_maruyama_step(self,xn,du,dt,dw):
        #takes one step forward in the euler maruyama discretisation
        return xn + du*dt + dw

    def generate_innovation(self,dt):
        return np.sqrt(dt)*torch.randn((self.batch_samples,1),requires_grad=False,dtype=torch.float32)


    def simulation_free(self,net,net_fixed):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural 
        - net_fixed: neural network with fixed parameters (copy)#TODO this could be replace with just detaching.
        ====

        """

        #set fixed network into evaluation mode
        net_fixed.eval()
        
        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)

        #set up a,abar,c,cbar
        a = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        abar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        #cbar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        #pdb.set_trace()
        t_vec = self.generate_time_grid()
        for i in range(0,self.t_steps-1):
            dt = (t_vec[i+1]-t_vec[i])
            dw = self.generate_innovation(dt)
            #dw_bar = dw.detach()#.numpy()

            du = net.forward(torch.column_stack((t_vec[i]*torch.ones(self.batch_samples,dtype=torch.float32)
                                                ,xinit)))
            du_bar = du.detach()#.numpy() #untracked copy

            a = a + dt*(torch.linalg.norm(du, axis=1) ** 2) 
                        #(torch.square(du))*dt
            c = c + np.sqrt(2)*torch.matmul(torch.unsqueeze(du, 1),
                                            torch.unsqueeze(dw, 2),
                                                ).reshape(self.batch_samples,) #du*dw

            with torch.no_grad(): #no gradient 
                
                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1) ** 2) 
                
            xinit = self.euler_maruyama_step(xinit,du_bar,dt,dw)
            
        #compute value of Lhat
        #see equation (10) and (11) in ref. 2410.05163
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + self.g(xinit)) * c))
        
        return Lhat, xinit #,loss_curr





if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
