from typing import Union

from transformers import PreTrainedTokenizerBase
from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataModule,
)


class CustomLanguageModelingDataModule(LanguageModelingDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = kwargs["max_length"]

    def tokenize_function(
        self,
        examples,
        tokenizer: Union[PreTrainedTokenizerBase],
        text_column_name: str = None,
    ):
        return tokenizer(
            examples[text_column_name],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

    @staticmethod
    def convert_to_features(examples, block_size: int, **kwargs):
        examples["labels"] = examples["input_ids"].copy()
        return examples
