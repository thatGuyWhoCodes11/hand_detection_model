import pandas as pd
csv_f=pd.read_csv("ArSL_dataset.csv")
csv_f=csv_f["class"].unique()
print(csv_f)
csv_f.to_csv("ArSL_dataset.csv",index=False,mode="w")