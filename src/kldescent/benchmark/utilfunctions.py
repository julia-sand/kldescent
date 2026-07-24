"""Useful functions used in computing perturbative solution"""

import csv

import numpy as np
from scipy.signal import savgol_filter

def bfun(bridge_type, Tf):

    if bridge_type=="marginal":
        return 1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf)
    elif bridge_type in ("half", "full"):
        return -(np.tanh(Tf/2)/Tf)*((1-np.exp(-Tf))**2)/(1+4*np.exp(-Tf)+np.exp(-2*Tf))
    else:
        return 1

def afun(bridge_type, Tf):

    if bridge_type=="marginal":
        return (Tf - 1 + np.exp(-Tf))/Tf
    elif bridge_type in ("half", "full"):
        return 1 + bfun(bridge_type,Tf) - 2*np.tanh(Tf/2)/Tf
    else:
        return 1

def savgol_filter_(y,dx,deriv=0):
    return savgol_filter(y, window_length=71, polyorder=3, deriv=deriv, delta=dx)

def apply_filter(y,filter_delta=None):
    """convolve array with box filter"""
    
    if filter_delta is None:
        #default to 1pc of the data vector
        filter_delta = int(np.floor(len(y) * 0.005))

    #apply convolution with moving averages
    out = np.convolve(y,(1/filter_delta)*np.ones(filter_delta),'same')
    
    return out
    #generic_filter(y,median,filter_delta,mode="constant")
    #out

def interpolate_(q,xs,phis):
    """Interpolate function phi from data points xs to q"""

    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx],left=0,right=0)

def normalise(x,y):
    idx = np.argsort(x)
    return y.copy()/np.trapezoid(y[idx],x[idx])

def zchop(x,tol=1e-100):
   
    return np.where(np.abs(x)<tol,0,x)

def make_csv_and_header(save_location,filename,header):
    #create the directory
    filename_full = save_location+"/"+filename+".csv"
    
    with open(filename_full,"w") as file: 
        writer = csv.writer(file,delimiter=" ",lineterminator="\n")
        writer.writerow(header)
    
    return filename_full

