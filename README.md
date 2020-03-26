
## TL;DR

A bottom-up refactor of (most of) the example scripts that will pave the way for more cool stuff to come in the future.

## In which order to read this proposal

```bash
# Clone the repo, then install torch and transformers
pip install transformers torch
```

1. The Datasets ([`datasets_glue.py`](datasets_glue.py) and [`datasets_lm.py`](datasets_lm.py)) are copy/paste of existing code.
	- they are currently `torch.data.Dataset`s but we could plug @thomwolf's framework agnostic datasets there.
1. The DataProcessors (in [`data_processor.py`](data_processor.py)): they possesses one or more `Dataset`s
    and are responsible for batching and pre-processing their samples
    when requested by the training loop.
	Name is TBD, ideas are welcome.
	In this proposal we have two of them:
	- DataProcessorForSequenceClassification
	- DataProcessorForLM
1. The [`trainer.py`](trainer.py) file contains:
	- `TrainingArgs` (read the docstring there)
	- the `Trainer` class
1. Finally, two working example scripts: [`run_glue.py`](run_glue.py) and [`run_language_modeling.py`](run_language_modeling.py)


## How does this articulate with pytorch-lightning examples

@srush has refactored a few examples into lightning and there seems to be a significant number of users who are using lightning/contributing other examples.

We do not want to have only lightning examples though, because
- we support Tensorflow (so we would have the same API in a TFTrainer, or just support TF transparently, like in `Pipeline`s).
- Lightning might be overkill or opaque for very simple training loops.

My thoughts:

- I was initially going to implement a `trainer.pl_train(**extra_pl_args)` on this `Trainer` prototype, that would have wrapper the model and its dataloader/etc. in a wrapper child of Pytorch-lightning and then just deferred to Lightning for everything.
- However, I'm now conflicted because:
	- the Trainer does not do anything in that class (just proxy and wrap)
	- it's using a slightly different API than lightning so it kind of obfuscates things for users of lightning. (If they're used to, and expect, the exact same method names etc. as encouraged by lightning)
- So maybe we can just have two different sets of examples, and see what happens "in the wild". In all cases, both implementations will be way more compact, because the `DataProcessor`s, the `TrainingArgs` above (and of course the datasets) can be shared.
- **Thoughts?**


