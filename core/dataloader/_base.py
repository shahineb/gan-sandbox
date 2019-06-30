import os
import sys
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

base_dir = os.path.join(os.path.dirname(os.getcwd()), "..")
sys.path.append(base_dir)
from utils.decorators import setseed


class BaseDataLoader(DataLoader, ABC):
    """Base class for all data loaders
    Inspired from https://github.com/victoresque/pytorch-template

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int)
        shuffle (bool)
        train_transform (callable): transformation for train set PIL image
        val_transform (callable): transformation for validation set PIL image
        sampler (Sampler): torch.utils.data.Sampler, defines dataset scope
        validation_split (int, float): number of validation samples or fraction in ]0, 1[
        num_workers (int)
        collate_fn (callable): merges a list of samples to form a mini-batch.
        n_steps (int): number of steps to take during an epoch (default: -1, iterates through dataset once for each epoch)
        seed (int): random seed

    Attributes:
        n_samples (int): size of dataset
        init_kwargs (dict): torch.utils.data.DataLoader init kwargs

    TODO : refactor transforms setting --> different from a dataset to the other, relou...
    """
    SEED = 73

    def __init__(self, dataset, batch_size, shuffle, train_transform,
                 num_workers, sampler, collate_fn, drop_last, n_steps=-1,
                 val_transform=None, seed=SEED, validation_split=0.):
        shuffle = False if sampler else shuffle
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.init_kwargs = {
            'dataset': self.set_transform(dataset, train_transform),
            'batch_size': batch_size,
            'shuffle': shuffle,
            'sampler': sampler,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last': drop_last
            }
        super(BaseDataLoader, self).__init__(**self.init_kwargs)
        self.validation_split = validation_split
        self.seed = seed or BaseDataLoader.SEED
        self.n_samples = len(dataset)
        self.n_steps = n_steps

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        """Number of iterations required to complete an epoch
        """
        if self.n_steps > 0:
            return self.n_steps
        else:
            return int(self.n_samples / self.batch_size)

    @setseed
    def choice(self, seed=None):
        """Returns random sample from dataset

        Args:
            seed (int): random seed
        """
        idx = np.random.randint(self.n_samples)
        return self.__getitem__(idx)

    @setseed
    def choices(self, k, seed=None):
        """returns random samples from dataset

        Args:
            k (int): number of samples to return
            seed (int): random seed
        """
        idxs = np.random.randint(0, self.n_samples, k)
        return [self.__getitem__(idx) for idx in idxs]

    def _split_sampler(self, split, scope):
        """Helper to build samplers for training and validation set

        Args:
            split (int, float): number of validation samples or fraction in ]0, 1[
            scope (np.ndarray): indexes on which sampling is to be performed
                default is on all samples
        """
        if split == 0:
            sampler_A = SubsetRandomSampler(scope)
            sampler_B = None
        else:
            np.random.seed(self.seed)
            np.random.shuffle(scope)

            if isinstance(split, int):
                assert split > 0
                assert split < len(scope), "subset size is configured to be larger than entire dataset."
                len_B = split
            else:
                len_B = int(len(scope) * split)

            idxB = scope[0:len_B]
            idxA = np.delete(scope, np.arange(0, len_B))

            sampler_A = SubsetRandomSampler(idxA)
            sampler_B = SubsetRandomSampler(idxB)

            # turn off shuffle option which is mutually exclusive with sampler
            self.shuffle = False
        return sampler_A, sampler_B

    def make_loader(self, sampler, **kwargs):
        """Builds dataloader for set described by sampler
        """
        kwargs = {**self.init_kwargs, **kwargs}
        kwargs.update({'sampler': sampler, 'shuffle': False})
        return DataLoader(**kwargs)

    @abstractmethod
    def set_transform(self, dataset, transform):
        """Generates a copy of a dataset setting its transform function

        Args:
            dataset (torch.utils.data.Dataset)
            transform (torchvision.transforms)
        """
        pass


class SelfSupervisedLoader(BaseDataLoader):
    """DataLoader for unsupervised training settings

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int)
        shuffle (bool)
        train_transform (callable): transformation for train set PIL image
        val_transform (callable): transformation for validation set PIL image
        validation_split (int, float): number of validation samples of fraction in ]0, 1[
        num_workers (int)
        sampler (Sampler): torch.utils.data.Sampler, defines dataset scope
        collate_fn (callable): merges a list of samples to form a mini-batch.
        seed (int): random seed

    Attributes:
        init_kwargs (dict): torch.utils.data.DataLoader init kwargs
        train_sampler (Sampler): training set sampler
        valid_sampler (Sampler): validation set sampler
        n_samples (int): size of dataset
        n_train_samples (int): size of training set
        n_val_samples (int): size of validation set
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split,
                 train_transform, val_transform, num_workers, sampler,
                 collate_fn, drop_last, seed):
        super(SelfSupervisedLoader, self).__init__(dataset=dataset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   validation_split=validation_split,
                                                   train_transform=train_transform,
                                                   val_transform=val_transform,
                                                   num_workers=num_workers,
                                                   sampler=sampler,
                                                   collate_fn=collate_fn,
                                                   drop_last=drop_last,
                                                   seed=seed)

        scope = self.sampler.indices if sampler else np.arange(self.n_samples)
        self.train_sampler, self.valid_sampler = self._split_sampler(split=self.validation_split,
                                                                     scope=scope)
        self.n_samples = len(self.train_sampler)
        self.n_train_samples = len(self.train_sampler)
        self.n_val_samples = len(self.valid_sampler) if self.valid_sampler else 0

    def validation_loader(self):
        dataset = self.set_transform(dataset=self.dataset, transform=self.val_transform)
        return self.make_loader(sampler=self.valid_sampler, dataset=dataset)
