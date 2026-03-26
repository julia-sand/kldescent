"""Useful functions used in computing perturbative corrections"""

import os

import numpy as np 

def interpolate_(q,xs,phis):
    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx],left=0,right=0)

def normalise(x,y):
    idx = np.argsort(x)
    return y.copy()/np.trapezoid(y[idx],x[idx])

def zchop(x,tol=1e-10):
   
    return np.where(np.abs(x)<tol,0,x)



def save_array(arr_out,save_location,filename,header):
    """saves an array as a csv to the given location
    args: 
        arr : ndarray
            array to save
        save_location : str
            filepath relative to where the code is run
        filename : str
            name of the file
        header : str 
            comment to add at the top of the csv
    """

    #make dir if does not exist
    os.makedirs(save_location,exist_ok=True)
    np.savetxt(save_location+"/"+filename+".csv", arr_out ,fmt="%10.6f",header = header,delimiter=',')
    return