import os
from os.path import expanduser

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast

from data_processor import DataProcessorForLM
from datasets_lm import LineByLineTextDataset
from trainer import Trainer, TrainingArgs

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=2,
    num_attention_heads=2,
)
model = RobertaForMaskedLM(config=config)

train_dataset = LineByLineTextDataset(tokenizer, os.path.expanduser("./data/README.md"))
eval_dataset = LineByLineTextDataset(tokenizer, os.path.expanduser("./data/LICENSE"))


processor = DataProcessorForLM(
    train_dataset,
    eval_dataset,
    tokenizer=tokenizer
)


training_args = TrainingArgs()

trainer = Trainer(
    model=model,
    data_processor=processor,
    args=training_args
)
trainer.train()


print()
