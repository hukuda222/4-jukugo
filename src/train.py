import warnings
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from factory.dataset import DataModule
from transformers import Trainer,TrainingArguments,AutoModelForCausalLM
from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataModule,
    LanguageModelingTransformer,
)
from transformers import AutoTokenizer


warnings.filterwarnings("ignore")

@hydra.main(config_path='../config', config_name='default_config')
def train(cfg: DictConfig) -> None:
    
    model = LanguageModelingTransformer(pretrained_model_name_or_path=cfg.model_checkpoint,
                                        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=cfg.model_checkpoint))

    dm = LanguageModelingDataModule(train_file= "dataset/train.csv",validation_file= "dataset/valid.csv",
                                    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=cfg.model_checkpoint))
    
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=5)
    trainer.fit(model, dm)

    

if __name__ == '__main__':
    train()