import pandas as pd
import regex
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")


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
    
with open("dataset/not-kanji-ids.csv","w") as f:
    for k,v in tokenizer.vocab.items():
        if not regex.search(r"^\p{Han}+$",k):
            f.write(str(v)+"\n")