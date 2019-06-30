# Loading data

We propose a Pytorch wrapping structure for data loading. Native Pytorch dataloaders are iterators built out of a dataset object which carries the transformations we want to apply on both input samples and targets, and a dataloader object describing the loading fashion (batch size, sampling, shuffling...). We here attempt to unify these structures under a single object.


## Initializing dataloader

Here's an example of a dataloader initialization

```python
# Load lib and setup global variables
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
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

# Show yielded samples
plt.figure(figsize=(16, 8))
grid = make_grid(inputs, nrow=8)
plt.imshow(grid.permute(1, 2, 0).numpy())
```
<p align="center"><img width="40%" src="https://github.com/shahineb/neural-conditioner/blob/master/docs/img/celeba_sample.png" /></p>

## Commands

__Dataloader attributes__

- Total number of samples :
```python
>>> dataloader.n_samples
162770
```

- Training set:
```python
>>> dataloader.n_train_samples  # Size
113939

>>> dataloader.train_sampler.indices  # Indices
array([ 98918,  39143, 129719, ...,  41281, 119107,  90258])
```

- Validation set:
```python
>>> dataloader.n_val_samples  # Size
48831

>>> dataloader.val_sampler.indices  # Indices
array([ 10473, 137015,  68392, ...,  81146, 104257,  41299])
```

- Batches:

```python
>>> dataloader.batch_size  # Size
8

>>> len(dataloader)  # Number of iterations per epoch
14242
```
