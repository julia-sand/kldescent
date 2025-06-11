import os
import pandas as pd

#quick script to join the two csvs with the cumulants data

currdir = os.path.dirname(__file__)

var = pd.DataFrame(pd.read_csv(currdir+"/variance.csv"))
mean = pd.DataFrame(pd.read_csv(currdir+"/mean.csv"))

df = pd.merge(var,mean,on="t")

#compute new column 
df["U2"] = df.Vpp*df.Ct/df.Qt + df.Vqp
df["u1"] = -df.U2*df.qt + (df.vp+df.Vqp*df.qt+df.Vpp*df.pt)

df.to_csv(currdir+"/cumulants.csv", index=False)