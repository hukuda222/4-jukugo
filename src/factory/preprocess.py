import pandas as pd

data=pd.read_csv("dataset/sanobo.csv")
data_sumitomo1=pd.read_csv("dataset/sumitomo1.csv")

data=data[~data["meaning"].isna()]

with open("dataset/train.txt","w") as f:
  for m,t in zip(data["meaning"],data["title"]):
    f.write(m.split("。")[0]+"</s>"+t.split("】")[0].replace("【","")[:4]+"\n")
  for m,t in zip(data_sumitomo1["meaning"],data_sumitomo1["title"]):
    f.write(m.split("。")[0]+"</s>"+t[:4]+"\n")
    
with open("dataset/valid.txt","w") as f:
  for m,t in zip(data["meaning"],data["title"]):
    f.write(m.split("。")[0]+"</s>"+t.split("】")[0].replace("【","")[:4]+"\n")
  for m,t in zip(data_sumitomo1["meaning"],data_sumitomo1["title"]):
    f.write(m.split("。")[0]+"</s>"+t[:4]+"\n")