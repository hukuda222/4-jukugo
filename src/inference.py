import torch
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer
import pandas as pd

tokenizer=AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")

model = LanguageModelingTransformer(
    pretrained_model_name_or_path="rinna/japanese-gpt2-medium",
    tokenizer=tokenizer,
    device_map="auto",
)
model = model.load_from_checkpoint("lightning_logs/version_5/checkpoints/epoch=9-step=1170.ckpt",
                                   device_map="auto").to(torch.device("cuda"))

bad_ids=pd.read_csv("dataset/not-kanji-ids.csv",header=None)[0]

src="早く寝て早く起きること"
input = tokenizer.encode(src, return_tensors="pt")
output = model.generate(src, device=torch.device("cuda"),
                        output_attentions=True,return_dict_in_generate=True,max_new_tokens=4,
                        bad_words_ids=[[bad_id] for bad_id in bad_ids if bad_id not in {0,2}],
                        no_repeat_ngram_size=1)

result=[]
for i in range(len(output[0][0])-len(input[0])):
    if output[0][0,len(input[0])+i]==2:
        continue
    if output[0][0,len(input[0])+i]==0:
        attn_score = torch.mean(output[1][i][0][0,:,0,:len(input[0])],axis=0)
        attn_score[0]=0
        attn_score[len(input[0])-1:]=0
        result+=tokenizer.batch_decode(input[:,torch.argmax(attn_score)])
    else:
        result+=tokenizer.batch_decode(output[0][:,len(input[0])+i])
print(result)
print("".join(result)[:4])