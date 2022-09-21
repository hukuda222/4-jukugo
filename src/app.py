import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

tokenizer=AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")

model = LanguageModelingTransformer(
    pretrained_model_name_or_path="rinna/japanese-gpt2-medium",
    tokenizer=tokenizer
)
model = model.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=9-step=1170.ckpt",
                                   device_map="auto").to(torch.device("cpu"))

bad_ids=pd.read_csv("dataset/not-kanji-ids.csv",header=None)[0]

def generate(src:str) -> str:
    input = tokenizer.encode(src, return_tensors="pt")
    output = model.generate(src, device=torch.device("cpu"),
                        output_attentions=True,return_dict_in_generate=True,max_new_tokens=4,
                        bad_words_ids=[[bad_id] for bad_id in bad_ids if bad_id not in {2}],
                        no_repeat_ngram_size=1)
    result=[]
    for i in range(len(output[0][0])-len(input[0])):
        #print(i,output[0][0,len(input[0])+i])
        if output[0][0,len(input[0])+i]==2:
            continue
        if output[0][0,len(input[0])+i]==0:
            attn_score = torch.mean(output[1][i][0][0,:,0,:len(input[0])],axis=0)
            attn_score[0]=0
            attn_score[len(input[0])-1:]=0
            result+=tokenizer.batch_decode(input[:,torch.argmax(attn_score)])
            #print(tokenizer.batch_decode(input[:,torch.argmax(attn_score)]))
        else:
            result+=tokenizer.batch_decode(output[0][:,len(input[0])+i])
    return "".join(result)[:4]
    
st.title('四字熟語ジェネレーター')

input = st.text_input('四字熟語の意味', '')
if st.button('生成'):
    st.write(generate(input))