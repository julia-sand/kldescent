import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv("src/refsolution/UnderdampedHalfNet/refsol_temp.csv"))

N=2000
x_ax = np.linspace(-6,6,num=N)

times_t = df["t"].unique()

#init new df
times_all = np.ones(N*len(times_t))
x_all = np.ones(N*len(times_t))
du_all = np.ones(N*len(times_t))

for t in times_t:
    du = df[df["t"]==t].U2*x_ax + df[df["t"]==t].u1
    new_data = [t*np.ones(N),x_ax,du]

#df.iloc[::5, :].to_csv("src/refsolution/UnderdampedHalfNet/refsol2.csv",index=False)