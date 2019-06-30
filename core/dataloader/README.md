# Loading data

We propose a Pytorch wrapping structure for data loading. Native Pytorch dataloaders are iterators built out of a dataset object which carries the transformations we want to apply on both input samples and targets, and a dataloader object describing the loading fashion (batch size, sampling, shuffling...). We here attempt to unify these structures under a single object.


## Initializing dataloader

Here's an example of a dataloader initialization

```python
# Load lib and setup global variables
import torchvision.transforms as transforms
data_dir = "./data/"

# Setup dataloader
dataloader = CelebALoader(data_dir=data_dir,
                          batch_size=8,
                          train_transform=transforms.ToTensor(),
                          val_transform=transforms.ToTensor(),
                          validation_split=0.3)

# Iterate over dataloader
iterator = iter(dataloader)
inputs = iterator.next()
```

## Commands

__Dataloader attributes__

- Total number of samples : `>>> dataloader.n_samples`

- Training set:
  - Size: `>>> dataloader.n_train_samples`
  - Indices: `>>> dataloader.train_sampler.indices`

- Validation set:
  - Size: `>>> dataloader.n_val_samples`
  - Indices: `>>> dataloader.valid_sampler.indices`
  - Get validation loader: `>>> val_loader = dataloader.validation_loader()`

- Batches:
  - Size: `>>> dataloader.batch_size`
  - Number of iterations per epoch: `>>> len(dataloader)`
