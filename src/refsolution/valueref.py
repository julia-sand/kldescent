"""
This file is used to compute a reference solution and deposits the results in a csv.

See Section 3. of my notes for how this is done. 

for comparison with 2411.08518, use beta = 2 

"""
import os
dirname = os.path.dirname(__file__)


import numpy as np 
import numpy.random as npr
from scipy.ndimage import generic_filter,median
import pdb

beta = 1

#final value function 
def vfinal(q):
    return (((q**2 - 1)**2) / 4) - ((q**2)/2)

def phifinal(q):
    return np.exp(-vfinal(q))

#final value function 
def dvfinal(q):
    return q**3 - (2*q)
    #q**3 - q

#final value function 
def dphifinal(q):
    return dvfinal(q)*phifinal(q)


mc_samples = 20000

Tf = 1#0.2
tstep = 0.05#0.005

##get a timegrid 
num_tsteps = int(np.floor(Tf/tstep)+1) #number of timesteps
t_vec = np.linspace(0,Tf,num_tsteps)
#pdb.set_trace()
# 

num_qsamples = 5000


qaxis = np.linspace(-4,4,num_qsamples)
qaxis_samples = np.concatenate([qaxis[np.newaxis,:]]*(num_tsteps-1))
qaxis_samples = np.concatenate([qaxis_samples[np.newaxis,:,:]]*(mc_samples))

#pdb.set_trace()

#generate trajectories
traj = qaxis_samples + np.cumsum(np.sqrt(2*tstep)*npr.randn(mc_samples, num_tsteps-1,num_qsamples),axis=1)
traj = np.flip(traj,axis = 1) #flip along columns

#find for t = T
arr = np.column_stack((Tf*np.ones(num_qsamples),qaxis,vfinal(qaxis),dvfinal(qaxis)))

for i in range(0,num_tsteps-1): 

    vout = -np.log(np.mean(phifinal(traj[:,i,:]),axis=0)).reshape(num_qsamples)
    
    #smooth
    #phi = generic_filter(phiout,median,50,mode="nearest")
    #generic_filter(np.log(phiout),median,50,mode="nearest")

    dvout = np.gradient(vout, qaxis)#generic_filter(,median,500,mode="nearest")
    
    arr = np.vstack((arr,np.column_stack((np.ones(num_qsamples)*t_vec[i],qaxis,vout,dvout))))

np.savetxt(os.path.join(dirname, "value_orig.csv"),arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')
print("done.")




