from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from transformers import PreTrainedModel, PreTrainedTokenizer

from data_processor import DataProcessor


"""
TrainingArgs would be the extraction of the argparse args we use in the example scripts today,
and that relate to the training loop itself.

We would have a way to turn those args automatically into argparse arguments to be able to 
specify them on the command line like today (I have a ten-liner class that does that).

(No special magic or third-party library needed)
"""

@dataclass
class TrainingArgs:
    per_gpu_train_batch_size: int = 4
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    no_cuda: bool = False
    n_gpu: int = 1

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    @property
    def train_batch_size(self) -> int:
        return self.per_gpu_train_batch_size * max(1, self.n_gpu)



"""
Trainer class: basically just the extraction of the training/eval loop
we have in example scripts today.

Note: this is very partial and we would add:
- gpu
- multi-gpu
- checkpointing
- eval
- mixed precision
"""


class Trainer:
    model: PreTrainedModel
    data_processor: DataProcessor
    args: TrainingArgs

    def __init__(self, model, data_processor, args):
        self.model = model
        self.data_processor = data_processor
        self.args = args
        self.optimizer = Adam(self.model.parameters())

    def collate(self, examples: List[torch.Tensor]):
        """
        Override in concrete subclass if necessary
        """
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_processor.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_processor.collate_batch,
        )

    def train(self):
        train_dataloader = self.get_train_dataloader()
        train_iterator = trange(0, int(self.args.num_train_epochs), desc="Epoch")
        global_step = 0
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                loss = self._training_step(batch)
                global_step += 1
                print(global_step, loss)

    def _training_step(self, batch) -> float:
        self.model.train()
        inputs = self.data_processor.preprocess_batch(batch)
        outputs = self.model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        return loss.item()




