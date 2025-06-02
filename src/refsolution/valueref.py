"""
This file is used to compute a reference solution given a Burger's equation
See Section 3. of my notes for how this is done. 

for comparison with 2411.08518, use set their beta = 2 

"""
import os

import numpy as np 

#import pdb
def integrate(a,b,T2f,vfinal,beta,dirname,
                tstep=0.005,
                mc_samples = 20000,
                num_qsamples = 5000):

    """
    Parameters:
    a - negative coefficient of second derivative in the burgers equation
    b - coefficient of the squared first derivative
    T2f - final time in the fast time
    vfinal - terminal condition

    mcsamples - number of MonteCarlo trajectories
    qsamples - number of points along q which are 
    """

    Generator = np.random.default_rng(seed=None)

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

    arr = np.column_stack((T2f*np.ones(num_qsamples),qaxis,vfinal(qaxis),np.gradient(vfinal(qaxis),qaxis)))
    for i in range(0,num_tsteps-1): 

        vout = -np.log(np.mean(phifinal(traj[:,i,:]),axis=0)).reshape(num_qsamples)
        
        dvout = np.gradient(vout, qaxis)
        arr = np.vstack((arr,np.column_stack((np.ones(num_qsamples)*t_vec[i],qaxis,vout,dvout))))

    os.makedirs(dirname, exist_ok = True) #add checkpoints folder
    np.savetxt(os.path.join(dirname, "value_orig.csv"),arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')
    print("done.")



if __name__=="__main__":

    dirname_curr = os.path.dirname(__file__)

    dirname = dirname_curr+f"/UnderdampedNoMomentum"

    Tf = 5
    epsilon = 0.2
    T2f = (epsilon**2)*Tf
    ell = np.sqrt(12.5)
    beta = 2

    a = -(Tf - 1 + np.exp(-Tf))/Tf
    b = -(1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf))

    def vfinal(q):
        return (((q*ell)**2-1)**2)/4
    integrate(a,b,T2f,
                vfinal,
                beta,
                dirname)


