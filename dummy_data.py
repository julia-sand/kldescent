##THIS FILE GENERATES SOME DUMMY TRAINING DATA
#emulates the perturbative drift csv. 

import pandas as pd 
import numpy as np 

Tf = 0.2


ts = np.linspace(0,0.2,11)
xs = np.linspace(-3,3,10)

T,X = np.meshgrid(ts,xs)

def dutest(t,x):
    return (t/Tf)*((x-1)**3) + ((Tf-t)/Tf)*(2*x*(x-1)**2)

DU = dutest(T,X)

#make new df with these
data= [np.hstack(T),np.hstack(X),np.hstack(DU)]
cols=["t","x","du"]

#append new
df = pd.DataFrame(dict(zip(cols, data)))


df.to_csv("kldescent/results1.csv",index=False)
