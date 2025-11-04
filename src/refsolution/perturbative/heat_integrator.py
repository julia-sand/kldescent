#integrate a one-dimensional heat equation 

def integrate(boundary,Tf2,
                num_tsteps = 100,
                mc_samples = 200,
                num_qsamples = 500,
                filter_delta = 10):
    """
    Parameters:
        boundary : class
            specifies the initial conditions 
        T2f : float
            final time in the slow time (Tf*epsilon**2)
        num_tsteps : float
            number of timesteps
        mcsamples : int
            number of MonteCarlo trajectories
        qsamples : int
            number of points along qaxis to evaluate the solution
        filter_delta : int 
            number of sample points to apply the convolution median smoothing filter for better gradients 
    Returns
        traj : array
            trajectories of the dynamics
    """

    Generator = np.random.default_rng(seed=42)

    def phifinal(q):
        return np.exp(-(b/a)*vfinal(q))

    ##get a timegrid 
    num_tsteps = int(np.floor(T2f/tstep)+1)
    t_vec = np.linspace(0,T2f,int(np.floor(T2f/tstep)+1))
   
    qaxis = np.linspace(-4,4,num_qsamples)
    qaxis_samples = np.concatenate([qaxis[np.newaxis,:]]*(num_tsteps-1))
    qaxis_samples = np.concatenate([qaxis_samples[np.newaxis,:,:]]*(mc_samples))

    #generate trajectories
    traj = qaxis_samples + \
        np.cumsum(np.sqrt(tstep)*Generator.standard_normal((mc_samples, num_tsteps-1,num_qsamples)),axis=1)
    #traj = np.flip(traj,axis = 1) #flip along columns

    return traj 

def save_traj(traj):
    """Saves the trajectories and values of v and dv 
    """

    arr_raw = np.column_stack((T2f*np.ones(num_qsamples),qaxis,vfinal(qaxis),np.gradient(vfinal(qaxis),qaxis)))
    arr = np.column_stack((T2f*np.ones(num_qsamples-(2*filter_delta)),qaxis[filter_delta:-filter_delta],vfinal(qaxis[filter_delta:-filter_delta]),dvfinal(qaxis[filter_delta:-filter_delta]))) #array with filtering applied
    
    for i in range(0,num_tsteps-1): 

        vout = -np.log(np.mean(phifinal(traj[:,i,:]),axis=0)).reshape(num_qsamples)
        
        dvout = np.gradient(vout, qaxis)
        arr_raw = np.vstack((arr_raw,np.column_stack((np.ones(num_qsamples)*t_vec[i],qaxis,vout,dvout))))
        
        vout_filter = generic_filter(vout,median,filter_delta,mode="nearest")
        dvout_filter = generic_filter(np.gradient(vout_filter[filter_delta:-filter_delta],qaxis[filter_delta:-filter_delta]),median,filter_delta,mode="nearest") 

        arr = np.vstack((arr,np.column_stack((np.ones(num_qsamples-(2*filter_delta))*t_vec[i],qaxis[filter_delta:-filter_delta],vout[filter_delta:-filter_delta],dvout_filter))))

    return arr_raw,arr 