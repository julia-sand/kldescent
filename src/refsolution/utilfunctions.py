import numpy as np 


def interpolate_(q,xs,phis):
    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx],left=0,right=0)

def normalise(x,y):
    idx = np.argsort(x)
    return y.copy()/np.trapz(y[idx],x[idx])

def zchop(x,tol=1e-10):
   
    return np.where(np.abs(x)<tol,0,x)