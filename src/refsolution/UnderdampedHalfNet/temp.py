import pandas as pd

df = pd.DataFrame(pd.read_csv("src/refsolution/UnderdampedHalfNet/refsol.csv"))

index_names = df[ df['q'] > 8 ].index

df.drop(index_names, inplace = True)


index_names = df[ df['q'] < -8 ].index

df.drop(index_names, inplace = True)



df.iloc[::5, :].to_csv("src/refsolution/UnderdampedHalfNet/refsol2.csv",index=False)