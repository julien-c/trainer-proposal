
## TL;DR

A bottom-up, incremental refactor of the example scripts (the main ones at least) that will pave the way for more cool stuff to come in the future.

**Timeline**: around ~2 weeks

## In which order to read the code in this proposal

```bash
# Clone the repo, then install torch and transformers
pip install transformers torch
```

1. The Datasets ([`datasets_glue.py`](datasets_glue.py) and [`datasets_lm.py`](datasets_lm.py)) are copy/paste of existing code.
	- they are currently `torch.data.Dataset`s but we will plug @thomwolf's framework agnostic datasets there.
1. The DataProcessors (in [`data_processor.py`](data_processor.py)) (name is T.B.D.): they possess one or more `Dataset`s
    and are responsible for batching and pre-processing their samples
    when requested by the training loop.
	Name is TBD, name ideas are welcome.
	In this proposal we have two of them:
	- DataProcessorForSequenceClassification
	- DataProcessorForLM
1. The [`trainer.py`](trainer.py) file contains:
	- `TrainingArgs` (read the docstring there)
	- the `Trainer` class (currently very partial, see docstring there.)
1. Finally, two working example scripts, to see the final "API" of a full training script: [`run_glue.py`](run_glue.py) and [`run_language_modeling.py`](run_language_modeling.py)


## How does this articulate with pytorch-lightning examples

@srush has refactored a few of the examples into lightning and there seems to be a fair number of users who are using lightning and contributing other examples.

We do not want to support only lightning examples though, because
- we support Tensorflow (so we should have about the same API in a `TFTrainer`, or even just support TF transparently, like in `Pipeline`s).
- Lightning might be overkill or opaque for very simple training loops.
- It's ok to explore different approaches at the same time.

**In all cases, this current proposal will also make the pytorch-lightning examples much cleaner and more compact**, because most of the heavy lifting is going to be in the `DataProcessor`s and the `TrainingArgs` above (and the datasets) and will be shared.

## Thoughts?


