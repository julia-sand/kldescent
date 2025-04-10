"""
This file is used to compute a reference solution for the probability distribution
See Section 3. of my notes for how this is done. 

for comparison with 2411.08518, use beta = 2 

"""
#get filenames
import os
dirname = os.path.dirname(__file__)



from sklearn.neighbors import KernelDensity
import numpy as np 
import numpy.random as npr
from scipy.ndimage import generic_filter

##get the drift 
data = np.loadtxt(os.path.join(dirname, "value.csv"),delimiter=",")

def du(time,q):
    duvals = data[np.where(data[:,0]==np.round(time,5))]
    return np.interp(q,duvals[:,1],duvals[:,-1])



#initial condition
def p_initial(q):
    return 0

mc_samples = 10000

##get an qaxis
num_qsamples = 5000
qinit = np.zeros(num_qsamples)
qaxis = np.linspace(-4,4,num_qsamples)

Tf = 0.2
tstep = 0.005

##get a timegrid 
num_tsteps = int(np.floor(Tf/tstep)+1) #number of timesteps
t_vec = np.linspace(0,Tf,num_tsteps)

########DISTRIBUTION W. GIRSANOV#########

#setup- make mc samples copies
qaxis_samples = np.concatenate([qinit[np.newaxis,:]]*(mc_samples))

arr = np.column_stack((np.zeros(num_qsamples),qaxis,np.zeros(num_qsamples)))

for t in range(num_tsteps-1):

    #sample a brownian motion
    currtime = t_vec[t]

    #sample brownian noise: W_{i-1} - W_i 
    innovation = npr.randn(mc_samples,num_qsamples)

    currdrift = du(currtime,qaxis_samples)#np.zeros(qaxis_samples.shape)

    #one step of reverse sde
    qaxis_samples = qaxis_samples + tstep*currdrift + np.sqrt(tstep)*innovation  #compute x_{t_{i-1}}
    
    ######
    #make a kde of the distribution 

    #run kde on these points
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.15).fit(qaxis_samples.reshape(-1, 1))

    #estimated pdf
    logrho_temp = kde.score_samples(qaxis.reshape(-1, 1))
    dens = np.exp(logrho_temp)

    arr = np.vstack((arr,np.column_stack((t_vec[t+1]*np.ones(num_qsamples),qaxis,dens))))

np.savetxt(os.path.join(dirname, "density.csv"),arr ,fmt="%10.6f",header = "t,q,rho",delimiter=',')
