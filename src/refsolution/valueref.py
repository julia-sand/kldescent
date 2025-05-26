"""
This file is used to compute a reference solution given a Burger's equation
See Section 3. of my notes for how this is done. 

for comparison with 2411.08518, use set their beta = 2 

"""
import os
dirname = os.path.dirname(__file__)

import numpy as np 
import numpy.random as npr
from scipy.ndimage import generic_filter,median


Generator = np.random.default_rng(seed=None)

#import pdb
def integrate(a,b,Tf,vfinal,
                mc_samples = 20,
                num_qsamples = 5):

    """
    Parameters:
    a - negative coefficient of second derivative in the burgers equation
    b - coefficient of the squared first derivative
    Tf - final time
    vfinal - terminal condition

    mcsamples - number of MonteCarlo trajectories
    qsamples - number of points along q which are 
    """

    def phifinal(q):
        return np.exp(-(b/a)*vfinal(q))


    Tf = 1#0.2
    tstep = 0.1

    ##get a timegrid 
    num_tsteps = int(np.floor(Tf/tstep)+1)
    t_vec = np.linspace(0,Tf,int(np.floor(Tf/tstep)+1))
   
    qaxis = np.linspace(-4,4,num_qsamples)
    qaxis_samples = np.concatenate([qaxis[np.newaxis,:]]*(num_tsteps-1))
    qaxis_samples = np.concatenate([qaxis_samples[np.newaxis,:,:]]*(mc_samples))

    #generate trajectories
    traj = qaxis_samples + \
        np.cumsum(np.sqrt(tstep)*Generator.standard_normal((mc_samples, num_tsteps-1,num_qsamples)),axis=1)
    #npr.randn(mc_samples, num_tsteps-1,num_qsamples)
    traj = np.flip(traj,axis = 1) #flip along columns

    arr = np.column_stack((Tf*np.ones(num_qsamples),qaxis,vfinal(qaxis)))#,dvfinal(qaxis)))
    for i in range(0,num_tsteps-1): 

        vout = -np.log(np.mean(phifinal(traj[:,i,:]),axis=0)).reshape(num_qsamples)
        
        #smooth
        #phi = generic_filter(phiout,median,50,mode="nearest")
        #generic_filter(np.log(phiout),median,50,mode="nearest")

        #dvout = np.gradient(vout, qaxis)#generic_filter(,median,500,mode="nearest")
        
        arr = np.vstack((arr,np.column_stack((np.ones(num_qsamples)*t_vec[i],qaxis,vout))))

    np.savetxt(os.path.join(dirname, "value_orig.csv"),arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')
    print("done.")

if __name__=="__main__":
    
    Tf = 1
    a = -(Tf - 1 + np.exp(-Tf))/Tf
    b = -(1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf))

    #final value function 
    def vfinal(q): # this is '-r'
        return (((q**2 - 1)**2) / 4) 

    integrate(a,b,Tf,vfinal)


