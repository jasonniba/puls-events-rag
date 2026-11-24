import pandas as pd

df = pd.read_csv(
    "data/raw/openagenda_api.csv",
    sep=";",
    low_memory=False
)

print(df.shape)
df.head()