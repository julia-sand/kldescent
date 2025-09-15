##THIS FILE GENERATES SOME DUMMY TRAINING DATA
#emulates the perturbative drift csv. 

import pandas as pd 
import numpy as np
 
N = 1

times_t = np.linspace(0,5,N)
xs = np.linspace(-8,8,N)

def dutest(t,x):
    return -x**3


#init new df
times_all = np.ones(N*len(times_t))
x_all = np.ones(N*len(times_t))
du_all = np.ones(N*len(times_t))

#pdb.set_trace()
for i in range(len(ts)):
    times_all[N*i:N*(i+1)] = np.ones(N)*times_t[i]
    x_all[N*i:N*(i+1)] = xs
    du_all[N*i:N*(i+1)] = dutest(times_t[i],xs)
    #new_data = [t*np.ones(N),x_ax,du]

df_out = pd.DataFrame(list(zip(times_all,x_all,du_all)),
                    columns=["t","x","du"])

#append new
df = pd.DataFrame(dict(zip(cols, data)))


df.to_csv("src/refsolution/UnderdampedHalfNet/test_result.csv",index=False)
