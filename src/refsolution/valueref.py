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


#final value function 
def vfinal(q):
    return (((q**2 - 1)**2) / 4)# - ((q**2)/2)

#final value function 
def dv(q):
    return q**3 - q

mc_samples = 10000

Tf = 0.2
tstep = 0.005

##get a timegrid 
num_tsteps = int(np.floor(Tf/tstep)+1) #number of timesteps
t_vec = np.linspace(0,Tf,num_tsteps)
#pdb.set_trace()

##get an qaxis
num_qsamples = 5000

#qaxis = np.array((num_qsamples,num_tsteps-1,mc_samples))

qaxis = np.linspace(-4,4,num_qsamples)
qaxis_samples = np.concatenate([qaxis[np.newaxis,:]]*(num_tsteps-1))
qaxis_samples = np.concatenate([qaxis_samples[np.newaxis,:,:]]*(mc_samples))

#generate backwards trajectories
traj = qaxis_samples - np.cumsum(np.sqrt(tstep)*npr.randn(mc_samples, num_tsteps-1,num_qsamples),axis=1)
traj = np.flip(traj,axis = 1) #flip along columns

#find for t = T
arr = np.column_stack((np.ones(num_qsamples)*Tf,qaxis,vfinal(qaxis),dv(qaxis)))

for i in range(0,num_tsteps-1): 
    print(t_vec[i])
    vout = np.mean(vfinal(traj[:,i-1,:]),axis=0)
    
    #smooth it 
    vout = generic_filter(vout,median,50,mode="nearest")
    duout = generic_filter(np.gradient(vout, qaxis),median,50,mode="nearest")
    #array out 
    
    arr = np.vstack((arr,np.column_stack((np.ones(num_qsamples)*t_vec[i],qaxis,vout,duout))))

np.savetxt(os.path.join(dirname, "value.csv"),arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')





