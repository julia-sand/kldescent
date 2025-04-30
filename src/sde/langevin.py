#this file computes the dynamics 
import numpy as np 
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

        return self.boundary.g(q)#torch.tensor(self.boundary.g(q),requires_grad=False,dtype=torch.float32)

    def euler_maruyama_step(self,xn,du,dt,dw,t):
        #takes one step forward in the euler maruyama discretisation
        bt = self.boundary.b(xn,t)#torch.tensor(self.boundary.b(xn,t),requires_grad=False,dtype=torch.float32)
        
        return xn + (bt + self.boundary.sigma0*du)*dt + self.boundary.sigma0*dw

    def get_ref_solution(self):
        #fetch and open the reference solution
        arr = np.loadtxt("./src/refsolution/value.csv",
                        delimiter=",")  
        return arr 

    def testing_time_grid(self,arr):
        #get an equally spaced time grid which can be used with the reference solution
        return np.unique(arr[:,0])

    def euler_maruyama_step_refsolution(self,arr,xn,currtime,dt,dw):
        #takes a step of the euler maruyama using the reference solution 

        #open ref solution
        #TODO add relative path
        
        dvtemp = arr[np.where(arr[:,0]==currtime),-1].reshape(5000) 
        qtemp = arr[np.where(arr[:,0]==currtime),1].reshape(5000) 
        
        ducurr = np.interp(xn.numpy(),qtemp,dvtemp,left=dvtemp[0],right=dvtemp[-1]).reshape((self.batch_samples,1))

        #takes one step forward in the euler maruyama discretisation
        return xn + torch.tensor(ducurr,dtype=torch.float32)*dt + dw

    def generate_innovation(self,dt):
        return np.sqrt(dt)*torch.randn((self.batch_samples,1),requires_grad=False,dtype=torch.float32)


    def simulation_free(self,net,logflag,logger):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network
        ====

        """
        
        #add as command line option??
        testing_mode = False
        
        #pdb.set_trace()
        t_vec = self.generate_time_grid()

        if testing_mode:
            arr = self.get_ref_solution()
            t_vec = self.testing_time_grid(arr)


        #initialise x values at \delta_0 (i.e. x=0)
        xinit = torch.tensor(self.boundary.generate_samples_q((self.batch_samples,1),"initial"),requires_grad=False,dtype=torch.float32)

        #set up a,abar,c,cbar
        a = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        abar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        bbar = torch.zeros((self.batch_samples,1),requires_grad=False,dtype=torch.float32)
        c = torch.zeros((self.batch_samples,1),dtype=torch.float32)
        
        #self.generate_time_grid()
        for i in range(0,self.t_steps-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt)

            du = net.forward(torch.column_stack((t_vec[i]*torch.ones(self.batch_samples,dtype=torch.float32)
                                                ,xinit)))
                   
            du_bar = du.detach()#.numpy() #untracked copy
            a = a + dt*(torch.linalg.norm(du, axis=1) ** 2) 
                        #(torch.square(du))*dt
            c = c + torch.matmul(torch.unsqueeze(du, 1),
                                            torch.unsqueeze(dw, 2),
                                                ).reshape(self.batch_samples,) #du*dw

            with torch.no_grad(): #no gradient 
                
                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1) ** 2) 
                bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                
            if testing_mode:
                xinit = self.euler_maruyama_step_refsolution(arr,xinit,t_vec[i],dt,dw)
            else: 
                xinit = self.euler_maruyama_step(xinit,du_bar,dt,dw,t_vec[i])
            
        #compute value of Lhat
        #see equation (10) and (11) in ref. 2410.05163
        Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + self.g(xinit)) * c))
        
        if logflag:
            logger.save_data(Lhat,xout.detach().cpu().numpy(),
                                net,epoch)

        return Lhat


if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
