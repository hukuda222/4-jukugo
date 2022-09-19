import torch
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

tokenizer=AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")

model = LanguageModelingTransformer(
    pretrained_model_name_or_path="rinna/japanese-gpt2-medium",
    tokenizer=tokenizer,
    device_map="auto",
)
model = model.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=9-step=1170.ckpt",
                                   device_map="auto").to(torch.device("cuda"))

input = tokenizer.encode("焼いた肉の定食</s>", return_tensors="pt")
output = model.generate("焼いた肉の定食</s>", device=torch.device("cuda"),
                        output_attentions=True,return_dict_in_generate=True,max_new_tokens=5)


result=[]
for i in range(len(output[0][0])-len(input[0])-1):
    if output[0][0,len(input[0])+i+1]==2:
        break
    if output[0][0,len(input[0])+i+1]==0:
        attn_score = torch.mean(output[1][i+1][0][0,:,0,:len(input[0])],axis=0)
        attn_score[0]=0
        attn_score[len(input[0])-2:]=0
        result+=tokenizer.batch_decode(input[:,torch.argmax(attn_score)])
    else:
        result+=tokenizer.batch_decode(output[0][:,len(input[0])+i+1])
print("".join(result)[:4])