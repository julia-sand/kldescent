
"""This file will smooth and compute gradient of the raw data from MC
simulations
"""
import numpy as np 

from scipy.ndimage import generic_filter,median

def preprocess(dirname,ell):
    """this is a script to preprocess the results of the reference solution calculation
    #we smooth the noise from the numerical calculation and multiply by the constant (-2)

    ====
    Parameters
    - ell: scaling factor of the numeric solution
    - dirname: location of raw results of integrating burger's equation
                and output location
    
    Out: 
    - csv containing the smoothed and differentiated results. 
    ====

    """
    refsolfile=dirname+"/value_orig.csv"
    
    ##fetch the perturbative underdamped approximation data
    arr = np.loadtxt(refsolfile,
                        delimiter=",")  

    arr[:,1] *= ell #rescale
    times = np.unique(arr[:,0])
    for t in range(len(times)):  
        filter_delta = 200
        valtemp = np.squeeze(arr[np.where(arr[:,0]==times[t]),-2])
        qtemp = np.squeeze(arr[np.where(arr[:,0]==times[t]),1])
        val_out = generic_filter(valtemp,median,filter_delta,mode="nearest")
        arr[np.where(arr[:,0]==times[t]),-2] = val_out
        dvout = generic_filter(np.gradient(val_out[filter_delta:-filter_delta],qtemp[filter_delta:-filter_delta]),median,filter_delta,mode="nearest") 
        arr[np.where(arr[:,0]==times[t]),-1] = -np.interp(qtemp,qtemp[filter_delta:-filter_delta],dvout,left=dvout[0],right=dvout[-1])

    np.savetxt(dirname+"/value.csv",arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')

    return 



if __name__=="__main__":
    ell = np.sqrt(12.5) #scaling factor for the q variable
    dirname = "src/refsolution/UnderdampedNoMomentum"

    preprocess(dirname,ell)