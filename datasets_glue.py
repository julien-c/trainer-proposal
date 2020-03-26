import logging
from typing import Any, Dict, List

import torch
from torch.utils.data.dataset import Dataset

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_processors
from transformers import InputFeatures
from transformers import PreTrainedTokenizer




"""
Toy example

(only supports MNLI, length is limited, some constants are hard-coded, etc.)
"""


class GlueMnliDataset(Dataset):
    features: List[InputFeatures]

    def __init__(self, tokenizer: PreTrainedTokenizer, evaluate=False):
        data_dir = "./data/MNLI"
        output_mode = "classification"
        processor = glue_processors["mnli"]()
        label_list = processor.get_labels()
        examples = (processor.get_dev_examples(data_dir) if evaluate else processor.get_dev_examples(data_dir))[:100]
        self.features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=tokenizer.max_len,
            output_mode=output_mode,
            pad_on_left=False,  # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
