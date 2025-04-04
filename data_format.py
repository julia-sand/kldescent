#temporary file used for any adhoc csv fixes. 

import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv("kldescent/results.csv",sep=","))

print(df.head(5))
#df[["t","x","UDdrift"]].to_csv("kldescent/results.csv",index=False)