"""
This file is used to compute a reference solution given a Burger's equation
See Section 3. of my notes for how this is done. 

for comparison with 2411.08518, use set their beta = 2 

"""

import os

import numpy as np 

def afun(Tf):
    return -(Tf - 1 + np.exp(-Tf))/Tf

def bfun(Tf):
    return -(1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf))

def integrate(a,b,T2f,vfinal,dvfinal,
                tstep=0.01,
                mc_samples = 200,
                num_qsamples = 500,
                filter_delta = 10):
    """
    Parameters:
        a : float
            negative coefficient of second derivative in the burgers equation
        b : float
            coefficient of the squared first derivative
        T2f : float
            final time in the slow time (Tf*epsilon**2)
        vfinal : function
            terminal condition
        dvfinal : function
            gradient of terminal condition
        tstep : float
            size of timestep in MC evolution
        mcsamples : int
            number of MonteCarlo trajectories
        qsamples : int
            number of points along qaxis to evaluate the solution
        filter_delta : int 
            number of sample points to apply the convolution median smoothing filter for better gradients 
    Returns
        value_orig.csv 
            csv file containing t,x,du
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
    #npr.randn(mc_samples, num_tsteps-1,num_qsamples)
    traj = np.flip(traj,axis = 1) #flip along columns

    arr_raw = np.column_stack((T2f*np.ones(num_qsamples),qaxis,vfinal(qaxis),np.gradient(vfinal(qaxis),qaxis)))
    arr = np.column_stack((T2f*np.ones(num_qsamples-(2*filter_delta)),qaxis[filter_delta:-filter_delta],vfinal(qaxis[filter_delta:-filter_delta]),np.gradient(vfinal(qaxis[filter_delta:-filter_delta]),qaxis[filter_delta:-filter_delta]))) #array with filtering applied
    
    for i in range(0,num_tsteps-1): 

        vout = -np.log(np.mean(phifinal(traj[:,i,:]),axis=0)).reshape(num_qsamples)
        
        dvout = np.gradient(vout, qaxis)
        arr_raw = np.vstack((arr_raw,np.column_stack((np.ones(num_qsamples)*t_vec[i],qaxis,vout,dvout))))
        
        vout_filter = generic_filter(vout,median,filter_delta,mode="nearest")
        dvout_filter = generic_filter(np.gradient(vout_filter[filter_delta:-filter_delta],qaxis),median,filter_delta,mode="nearest") 
        # = -np.interp(qaxis,qaxis[filter_delta:-filter_delta],dvout,left=dvout[0],right=dvout[-1])
        arr = np.vstack((arr,np.column_stack((np.ones(num_qsamples-(2*filter_delta))*t_vec[i],qaxis[filter_delta:-filter_delta],vout[filter_delta:-filter_delta],dvout))))

    return arr_raw,arr 


def save_file(arr,dirname,filename):
    os.makedirs(dirname, exist_ok = True) #add checkpoints folder
    np.savetxt(os.path.join(dirname, filename), arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')
    print("done.")


if __name__=="__main__":

    dirname_curr = os.path.dirname(__file__)

    dirname = dirname_curr+f"/UnderdampedNoMomentum"

    Tf = 5
    epsilon = 0.2
    T2f = (epsilon**2)*Tf
    
    a = afun(Tf)
    b = bfun(Tf)

    def vfinal(q):
        return ((q**2-1)**2)/4
    
    def dvfinal(q):
        return q*(q**2-1)
        
    arr_raw, arr = integrate(a,b,T2f,
                vfinal,dvfinal)
    save_file(arr,dirname,"value.csv")
    save_file(arr_raw,dirname,"value_orig.csv")

