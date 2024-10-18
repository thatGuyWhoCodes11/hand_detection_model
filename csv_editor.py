import pandas as pd
csv_f=pd.read_csv("new_coords.csv")
csv_f=csv_f[~(csv_f["class"]=="ta")]
csv_f.to_csv("new_coords.csv",index=False,mode="w")