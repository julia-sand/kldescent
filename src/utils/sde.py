#this file computes the dynamics 
import numpy as np 
import numpy.random as npr
import torch
import utils.boundary as boundary

def sde_overdamped(xinit,net,t_vec,batch_samples,epsilon,epoch,logfile):
    """
    Input: 
    -net    neural network current state 
    -t_vec  vector of times 
    -batch_samples  number of samples in the batch
    -epsilon    parameter of the dynamics

    Output: 
    -estimated cost under current parameters
    -kl divergence between current data 
    """
    net.eval() #evaluation mode

    
    with torch.no_grad(): #this disables gradient computation
        loss_curr = 0 
        t_steps = len(t_vec)
        for i in range(0,t_steps-1):
            du = net(torch.column_stack((t_vec[i]*torch.ones(batch_samples,dtype=torch.float32)
                                                ,xinit)))
                
            #go forward one step 
            xinit = xinit + (du)*(t_vec[i+1]-t_vec[i]) + np.sqrt(2*(t_vec[i+1]-t_vec[i]))*torch.randn((batch_samples,1))


        #add to loss integral
        loss_curr += torch.sum(du**2)*(t_vec[i+1]-t_vec[i])

    #compute kl between target assigned distribution 
    
    #log the loss function metrics
    logfile.log([epoch,losscurr.item(),kl.kl_div(xout)])
    
    return xinit#,loss_curr


def sde_underdamped(xinit,net,t_vec,batch_samples,epsilon):
    """
    Input: 
    -net    neural network current state 
    -t_vec  vector of times 
    -batch_samples  number of samples in the batch
    -epsilon    parameter of the dynamics

    Output: 
    -estimated cost under current parameters
    -kl divergence between current data 
    """
    
    loss_curr = 0 
    t_steps = len(t_vec)
    for i in range(0,t_steps-1):
        with torch.no_grad(): #this disables gradient computation
            du = net(torch.column_stack((t_vec[i]*torch.ones(batch_samples,dtype=torch.float32)
                                            ,xinit[:,0])))
            
        #go forward one step 
        xinit[:,0:0] = xinit[:,0:0] + epsilon*xinit[:,1:1]*(t_vec[i+1]-t_vec[i])
        xinit[:,1:1] = xinit[:,1:1] + -(xinit[:,0:0] + epsilon*du)*(t_vec[i+1]-t_vec[i]) + np.sqrt(2*(t_vec[i+1]-t_vec[i]))*torch.randn((batch_samples,1))


        #add to loss integral
        loss_curr += torch.sum(du**2)*(t_vec[i+1]-t_vec[i])

    #compute kl between target assigned distribution 
    
    return xinit,loss_curr


class SDEOverdamped():
    def __init__(self,batch_samples,epsilon,Tf,t_steps):
        self.batch_samples = batch_samples # number of samples in each step 
        self.epsilon = epsilon 
        self.Tf = Tf #final time
        self.t_steps = t_steps #number of timesteps in discretisation

    def generate_time_grid(self):

        #samples the interval [0,Tf] and returns sorted set of values to be used in integration

        return np.concatenate([[0],
                                np.sort(np.random.uniform(0, self.Tf, self.t_steps-1)),
                                [self.Tf]])


    def g(self,q): 
        #this is the penalty for the final time

        #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

        return -(-torch.log(boundary.q_marginal_final(q)) - (q**2)/2)

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
                #du_bar = net_fixed(torch.column_stack((self.t_vec[i]*torch.ones(arrayshape[0],dtype=torch.float32)
                #                                ,xinit)))
                abar = abar + (1/2)*(torch.square(du))*dt
                #go forward one step 
                xinit = self.em_step(xinit,du,dt,dw)
                #xinit - du_bar*dt + np.sqrt(2)*dw

        #compute value of Lhat
        #see equation (10) and (11) in ref. 2410.05163
        Lhat = a +(abar+self.g(xinit))*c
        #Lhat.mean().backward()

        return Lhat, xinit #,loss_curr


    def get_net_output(self,net,net_fixed,plot_times):
        #computes neural network output (forward pass)
        # for selected times
        
        dus = []
        for t in plot_times:
            duplot = net(torch.column_stack((t*torch.ones(self.batch_samples,dtype=torch.float32)
                                                ,torch.linspace(-4,4,self.batch_samples)))).detach().cpu().numpy()
            duplot_fixed = net_fixed(torch.column_stack((t*torch.ones(self.batch_samples,dtype=torch.float32)
                                                ,torch.linspace(-4,4,self.batch_samples)))).detach().cpu().numpy()
            dus.append([duplot,duplot_fixed])  
        return dus



if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
