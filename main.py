#%%
import pandas as pd
import pandas_dedupe

df = pd.read_csv("result7.csv")
df["latlong"] = df[['lat','long']].apply(tuple,axis=1)
#%%
df = pandas_dedupe.dedupe_dataframe(df,[("latlong","LatLong")],  sample_size=1)
