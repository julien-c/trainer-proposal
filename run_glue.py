from data_processor import DataProcessorForSequenceClassification
from datasets_glue import GlueMnliDataset
from trainer import Trainer, TrainingArgs
from transformers import RobertaConfig
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizerFast


tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_labels=3,
)
model = RobertaForSequenceClassification(config=config)

train_dataset = GlueMnliDataset(tokenizer=tokenizer)
eval_dataset = GlueMnliDataset(tokenizer=tokenizer, evaluate=True)


processor = DataProcessorForSequenceClassification(
    train_dataset,
    eval_dataset,
)

training_args = TrainingArgs()



trainer = Trainer(
    model=model,
    data_processor=processor,
    args=training_args
)
trainer.train()


print()
