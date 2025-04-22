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

    return -(-torch.log(boundary.q_marginal_final(q)) - (q**2)/2)


def f(q): 
    #this is the penalty for the final time

    #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

    return 0

def simulation_free_overdamped(net,
                                net_fixed,
                                t_vec,
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
    batch_samples = 8000 # number of samples in each step 

    #set fixed network into evaluation mode
    net_fixed.eval()
    
    loss_curr = 0 
    t_steps = len(t_vec)
    
    arrayshape = (batch_samples,1)
    xinit = torch.zeros(arrayshape)

    #set up a,abar,c 
    a = torch.zeros(arrayshape)
    abar = torch.zeros(arrayshape)
    c = torch.zeros(arrayshape)

    #get ref data
    #import os 
    #dirname = os.path.dirname(__file__) #fetches current location 
    #refsolfile=dirname+"/../refsolution/value.csv"
    
    ##fetch the perturbative underdamped approximation data
    #arr = np.loadtxt(refsolfile,delimiter=",")  
    
    for i in range(0,t_steps-1):
        #print(t_vec[i])
        dt = (t_vec[i+1]-t_vec[i])
        dw = np.sqrt(dt)*torch.randn(arrayshape)
        
        du = 2*net(torch.column_stack((t_vec[i]*torch.ones(arrayshape[0],dtype=torch.float32)
                                            ,xinit)))
        a = a + (1/2)*(torch.square(du))*dt
        c = c - du*dw #np.sqrt(2)*du*dw

        with torch.no_grad(): #no gradient 
            du_bar = 2*net_fixed(torch.column_stack((t_vec[i]*torch.ones(arrayshape[0],dtype=torch.float32)
                                            ,xinit)))
            
            #get dv
            #arrtemp = arr[np.where(arr[:,0]==np.round(t_vec[i],6)),:]
            #arrtemp = np.squeeze(arrtemp)
            
            #du_bar = np.interp(xinit.numpy(),arrtemp[:,1],arrtemp[:,-1],left=arrtemp[0,-1],right=arrtemp[-1,-1])
            #du_bar = torch.from_numpy(du_bar).reshape(xinit.shape)

            abar = abar + (1/2)*(torch.square(du_bar))*dt

            #go forward one step 
            xinit = xinit - du_bar*dt + np.sqrt(2)*dw

    #compute value of Lhat
    #see equation (10) and (11) in ref. 2410.05163
    Lhat = a +(abar+g(xinit))*c
    #Lhat.mean().backward()

    return Lhat, xinit #,loss_curr




if __name__=="__main__":
    sde_evolve(net,t_vec,batch_samples,epsilon)
