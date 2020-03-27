My thoughts:

- I was initially going to implement a stub for a `trainer.pl_train(**extra_pl_args)` on this `Trainer` prototype, that would have wrapped the model and its dataloader/etc. in a wrapper child of Pytorch-lightning's LightningModule and then just deferred to Lightning for everything.
- However, I'm now conflicted because:
	- the Trainer would not do anything in that case (just proxy and wrap), which is kinda deceptive.
	- It would then expose a slightly different API than lightning so it kind of obfuscates things for users of lightning. (They are used to, and expect, the exact same method names etc. as encouraged by lightning)
- So maybe we can just have two different sets of examples, and see what happens "in the wild". In all cases, both implementations will be way more compact, because most of the heavy lifting is going to be in the `DataProcessor`s and the `TrainingArgs` above (and of course the datasets) and would be shared.

