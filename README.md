
## TL;DR

A bottom-up refactor of (most of) the example scripts that will pave the way for more cool stuff to come in the future.

## In which order to read this proposal

```bash
# Clone the repo, then install torch and transformers
pip install transformers torch
```

1. The Datasets (`dataset_glue.py` and `dataset_lm.py`) are copy/paste of existing code.
	- they are currently `torch.data.Dataset`s but we could plug @thomwolf's framework agnostic datasets there.
1. The DataProcessors (in [`data_processor.py`](data_processor.py)): they possesses one or more `Dataset`s
    and are responsible for batching and pre-processing their samples
    when requested by the training loop.
	Name is TBD, ideas are welcome.
	In this proposal we have two of them:
	- DataProcessorForSequenceClassification
	- DataProcessorForLM
1. The `trainer.py` file contains:
	- `TrainingArgs` (read the docstring there)
	- the `Trainer` class
1. Finally, two working example scripts: `run_glue.py` and `run_language_modeling.py`


