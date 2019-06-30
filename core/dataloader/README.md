# Loading data

We propose a Pytorch wrapping structure for data loading. Native Pytorch dataloaders are iterators built out of a dataset object which carries the transformations we want to apply on both input samples and targets, and a dataloader object describing the loading fashion (batch size, sampling, shuffling...). We here attempt to unify these structures under a single object.
