##UNDERDAMPED DYNAMICS GO HERE


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
