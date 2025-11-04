"""
This file is used to compute a reference solution given a Burger's equation
See Section 3. of my notes for how this is done. 

for comparison with 2411.08518, use set their beta = 2 

"""

import os

import numpy as np 
from scipy.ndimage import generic_filter,median

def save_file(arr,dirname,filename):
    os.makedirs(dirname, exist_ok = True) #add checkpoints folder
    np.savetxt(os.path.join(dirname, filename), arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')
    print("done.")

if __name__=="__main__":

    dirname_curr = os.path.dirname(__file__)

    dirname = dirname_curr+"/results"

    Tf = 5
    epsilon = 0.2
    T2f = (epsilon**2)*Tf
    
    a = afun(Tf)
    b = bfun(Tf)

    def vfinal(q):
        return ((q**2-1)**2)/4
    
    def dvfinal(q):
        return q*(q**2-1)
        
    traj_final = integrate(a,b,T2f,
                vfinal,dvfinal)
    save_file(arr,dirname,"value.csv")
    save_file(arr_raw,dirname,"value_orig.csv")

