from typing import Any

import torch
import transformers
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_transformers.core import TaskTransformer


class LanguageModelingTransformer(TaskTransformer):
    
    def generate(self, text: str, device: torch.device = torch.device("cpu"), **kwargs) -> Any:
        if self.tokenizer is None:
            raise MisconfigurationException(
                "A tokenizer is required to use the `generate` function. "
                "Please pass a tokenizer `LanguageModelingTransformer(tokenizer=...)`."
            )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        return self.model.generate(inputs["input_ids"], **kwargs)