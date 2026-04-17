"""Useful functions used in computing perturbative solution"""

import os
import csv

import numpy as np

def bfun(bridge_type, Tf):
    
    if bridge_type=="marginal":
        return 1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf)
    elif bridge_type =="half":
        return -(np.tanh(Tf/2)/Tf)*((1-np.exp(-Tf))**2)/(1+4*np.exp(-Tf)+np.exp(-2*Tf))
    else:
        return 1 

def afun(bridge_type,Tf):
    
    if bridge_type=="marginal":
        return (Tf - 1 + np.exp(-Tf))/Tf
    elif bridge_type =="half":
        return 1 + bfun(bridge_type,Tf) - 2*np.tanh(Tf/2)/Tf #-(-1 + (2*np.tanh(Tf/2)/Tf)-self.B)  
    else:
        return 1 

def apply_filter(y,filter_delta=None):
    """convolve array with box filter"""
    
    if filter_delta is None:
        #default to 5pc of the data vector
        filter_delta = int(np.floor(len(y)*0.05))

    #apply convolution with moving averages
    out = np.convolve(y,(1/filter_delta)*np.ones(filter_delta),'same')
    
    return out

def append_array_to_csv(filename,data):
    """append numpy array to a csv"""
    
    with open(filename,"a") as file:
      np.savetxt(file,data,fmt="%10.6f",delimiter=',')
  
def interpolate_(q,xs,phis):
    """Interpolate function phi from data points xs to q"""

    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx],left=0,right=0)

def normalise(x,y):
    idx = np.argsort(x)
    return y.copy()/np.trapezoid(y[idx],x[idx])

def zchop(x,tol=1e-10):
   
    return np.where(np.abs(x)<tol,0,x)

def make_csv_and_header(save_location,filename,header):
    #create the directory
    filename_full = save_location+"/"+filename+".csv"
    
    with open(filename_full,"w") as file: 
        writer = csv.writer(file,delimiter=" ",lineterminator="\n")
        writer.writerow(header)
    
    return filename_full

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

