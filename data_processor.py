from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from datasets_lm import mask_tokens
from transformers import InputFeatures, PreTrainedTokenizer




@dataclass
class DataProcessor(ABC):
    """
    A `DataProcessor` (name T.B.D.) possesses one or more `Dataset`s
    and is responsible for batching and pre-processing their samples
    when requested by the training loop.
    """
    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None

    @abstractmethod
    def collate_batch(self):
        """
        Take a list of samples from a Dataset and collate them into a batch.
        """
        pass

    def preprocess_batch(self, batch) -> Dict[str, torch.Tensor]:
        """
        Take a batch, and return a dict of model inputs.

        Default implementation is identity.
        """
        return batch



@dataclass
class DataProcessorForSequenceClassification(DataProcessor):
    output_mode = "classification"

    def collate_batch(self, features: List[InputFeatures]) -> Dict[str, torch.Tensor]:
        if self.output_mode == "classification":
            labels = torch.tensor([f.label for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label for f in features], dtype=torch.float)
        return {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            "token_type_ids": torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            "labels": labels
        }


@dataclass
class DataProcessorForLM(DataProcessor):
    tokenizer: Optional[PreTrainedTokenizer] = None
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(
            examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

    def preprocess_batch(self, batch) -> Dict[str, torch.Tensor]:
        inputs, labels = mask_tokens(batch, self.tokenizer, self)
        return {"input_ids": inputs, "masked_lm_labels": labels}
