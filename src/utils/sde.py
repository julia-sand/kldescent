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

def g(q): 
    #this is the penalty for the final time

    #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

    return -torch.log(boundary.q_marginal_final(q)) - (q**2)/2


def f(q): 
    #this is the penalty for the final time

    #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

    return 0

def simulation_free_overdamped(xinit,
                                net,
                                net_fixed,
                                t_vec,
                                batch_samples,
                                epsilon):
    """
    Following 2410.05163 (see proposition 3). 

    ====
    Parameters
    - xinit,
    - net: neural 
    - net_fixed: neural network with fixed parameters (copy)
    - t_vec: vector of time coordinates
    - batch_samples: number of samples in the batch
    - epsilon: dimensionless parameter
    ====


    """

    #set fixed network into evaluation mode
    net_fixed.eval()
    
    loss_curr = 0 
    t_steps = len(t_vec)

    #set up a,abar,c 
    a = torch.zeros((batch_samples,1))
    abar = torch.zeros((batch_samples,1))
    c = torch.zeros((batch_samples,1))

    

    for i in range(0,t_steps-1):

        dt = (t_vec[i+1]-t_vec[i])
        dw = np.sqrt(dt)*torch.randn((batch_samples,1))
        
        du = net(torch.column_stack((t_vec[i]*torch.ones(batch_samples,dtype=torch.float32)
                                            ,xinit)))
        a = a + (1/2)*(torch.square(du))*dt
        c = c + du*dw #np.sqrt(2)*du*dw

        with torch.no_grad(): #no gradient 
            du_bar = net_fixed(torch.column_stack((t_vec[i]*torch.ones(batch_samples,dtype=torch.float32)
                                            ,xinit)))

            abar = abar + (1/2)*(torch.square(du_bar))*dt
            #bbar = bbar + f(xinit)*dt

            #go forward one step 
            xinit = xinit + (du_bar)*dt + np.sqrt(2)*dw

    #compute value of Lhat
    #see equation (10) and (11) in ref. 2410.05163
    Lhat = a +(abar+g(xinit))*c
    #Lhat.mean().backward()

    #print(Lhat)

    return Lhat, xinit #,loss_curr




if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
