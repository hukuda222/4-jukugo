import warnings
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingTransformer,
)
from transformers import AutoTokenizer
from factory.dataset import CustomLanguageModelingDataModule


warnings.filterwarnings("ignore")


@hydra.main(config_path="../", config_name="config")
def train(cfg: DictConfig) -> None:

    model = LanguageModelingTransformer(
        pretrained_model_name_or_path=cfg.model_checkpoint,
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=cfg.model_checkpoint
        ),
    )

    dm = CustomLanguageModelingDataModule(
        train_file=cfg.train_path,
        validation_file=cfg.valid_path,
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=cfg.model_checkpoint
        ),
        max_length=cfg.max_length,
    )

    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=cfg.max_epoch)
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
