import pytorch_lightning as pl
from transformers import AutoTokenizer
from datasets import load_dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.max_length=cfg.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint)
        self.dataset = load_dataset("text", data_files={"train": "../dataset/train.txt", "validation": "../dataset/valid.txt"})

    def _tokenize(self, x):
        result = self.tokenizer(x["text"],max_length=56,padding='max_length',truncation=True)
        result["labels"]=result["input_ids"].copy()
        return result

    def setup(self, stage=None):
        self.lm_dataset = self.dataset.map(self._tokenize, batched=True, num_proc=4, remove_columns=["text"])

    def train_dataloader(self):
        return self.lm_dataset["train"]

    def val_dataloader(self):
        return self.lm_dataset["validation"]
    
    


