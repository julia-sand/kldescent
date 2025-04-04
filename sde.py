#this file computes the dynamics 
import numpy as np 
import numpy.random as npr
#import model
from kl import kl_div
import torch
#import viz.plots as plots

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
            xinit = xinit + -(du)*(t_vec[i+1]-t_vec[i]) + np.sqrt(2*(t_vec[i+1]-t_vec[i]))*torch.randn((batch_samples,1))


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

def g(x):
    #this is the penalty for the final time

    return x

def simulation_free_overdamped(xinit,net,net_fixed,t_vec,batch_samples,epsilon):
    """
    2410.05163
    """

    #set fixed network into evaluation mode
    net_fixed.eval()
    
    loss_curr = 0 
    t_steps = len(t_vec)
    #with torch.no_grad(): #this disables gradient computation
    for i in range(0,t_steps-1):
        du = net1(torch.column_stack((t_vec[i]*torch.ones(batch_samples,dtype=torch.float32)
                                            ,xinit)))

        with torch.no_grad():
            du_bar = net_fixed(torch.column_stack((t_vec[i]*torch.ones(batch_samples,dtype=torch.float32)
                                            ,xinit)))

        dt = (t_vec[i+1]-t_vec[i])
        dw = np.sqrt(dt)*torch.randn((batch_samples,1))

        #go forward one step 
        xinit = xinit + du_bar*dt + np.sqrt(2)*dw

        a = a + (1/2)*(torch.square(du))*dt
        abar = abar + (1/2)*(torch.square(du_bar))*dt
        c = c + np.sqrt(2)*du*dw

        #add to loss integral
        #loss_curr += torch.sum(du**2)*(t_vec[i+1]-t_vec[i])

    #compute value of Lhat
    Lhat = torch.mean(a +(abar+g(xinit))*c)
    logger.log()
    return xinit#,loss_curr




if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
