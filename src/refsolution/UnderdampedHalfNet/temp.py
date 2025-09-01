import pandas as pd
import numpy as np

import pdb
#script to prepare some test training data 
df = pd.DataFrame(pd.read_csv("src/refsolution/UnderdampedHalfNet/refsol_temp.csv",header=0))

N=2000
x_ax = np.linspace(-6,6,num=N)

times_t = df["t"].unique()[::4] 

#init new df
times_all = np.ones(N*len(times_t))
x_all = np.ones(N*len(times_t))
du_all = np.ones(N*len(times_t))

#pdb.set_trace()
for i in range(len(times_t)):
    times_all[N*i:N*(i+1)] = np.ones(N)*times_t[i]
    x_all[N*i:N*(i+1)] = x_ax
    du_all[N*i:N*(i+1)] = df[df["t"]==times_t[i]].U2.values*x_ax + df[df["t"]==times_t[i]].u1.values
    
    #new_data = [t*np.ones(N),x_ax,du]

df_out = pd.DataFrame(list(zip(times_all,x_all,du_all)),
                    columns=["t","x","du"])

#df_out.to_csv("src/refsolution/UnderdampedHalfNet/refsol_temp2.csv",index=False)
df_out.iloc[::10, :].to_csv("src/refsolution/UnderdampedHalfNet/refsol_temp2.csv",index=False)