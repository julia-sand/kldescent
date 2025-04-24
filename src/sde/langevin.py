#this file computes the dynamics 
import numpy as np 
import numpy.random as npr
import torch
from sde.boundary import AssignedBoundaryData


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

        return -(-torch.log(self.boundary.q_marginal_final(q)) - (q**2)/2)

    def em_step(self,xn,du,dt,dw):
        #takes one step forward in the euler maruyama discretisation
        return xn + du*dt + np.sqrt(2)*dw

    def simulation_free(self,net,net_fixed):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - xinit,
        - net: neural 
        - net_fixed: neural network with fixed parameters (copy)
        ====

        """

        #set fixed network into evaluation mode
        #TODO add temporary network here???
        net_fixed.eval()
        
        
        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.zeros((self.batch_samples,1))

        #set up a,abar,c 
        a = torch.zeros((self.batch_samples,1))
        abar = torch.zeros((self.batch_samples,1))
        c = torch.zeros((self.batch_samples,1))

        t_vec = self.generate_time_grid()
        for i in range(0,self.t_steps-1):
            dt = (t_vec[i+1]-t_vec[i])
            dw = np.sqrt(dt)*torch.randn((self.batch_samples,1))
            
            du = net(torch.column_stack((t_vec[i]*torch.ones(self.batch_samples,dtype=torch.float32)
                                                ,xinit)))
            a = a + (1/2)*(torch.square(du))*dt
            c = c + du*dw #np.sqrt(2)*du*dw

            with torch.no_grad(): #no gradient 
                
                abar = abar + (1/2)*(torch.square(du))*dt
                #go forward one step 
                xinit = self.em_step(xinit,du,dt,dw)
                #xinit - du_bar*dt + np.sqrt(2)*dw

        #compute value of Lhat
        #see equation (10) and (11) in ref. 2410.05163
        Lhat = a + (abar+self.g(xinit))*c
        #Lhat.mean().backward()

        return Lhat, xinit #,loss_curr





if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
