import copy
from torchvision.datasets import CelebA
from torch.utils.data.dataloader import default_collate
from ._base import SelfSupervisedLoader
from .utils import collate


class CelebALoader(SelfSupervisedLoader):
    """Self-supervised dataloader for CelebA dataset

    Args:
        data_dir (str): dataset root directory
        batch_size (int): how many samples per batch to load (default: 1)
        split (str): in {'train', 'valid', 'test', 'all'}.
        train_transform (callable): transformation for train set PIL image
        val_transform (callable): transformation for validation set PIL image
        shuffle (bool):
        validation_split (int, float): number of validation samples of fraction in ]0, 1[
        num_workers (int)
        collate_fn (callable): merges a list of samples to form a mini-batch.
        download (bool): if True, dataset is downloaded and unzipped in data_dir
    """
    def __init__(self, data_dir, batch_size=1, split='train', train_transform=None,
                 val_transform=None, shuffle=True, validation_split=0.,
                 num_workers=1, collate_fn=collate.drop_target, download=False,
                 drop_last=True, sampler=None, seed=SelfSupervisedLoader.SEED):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = CelebA(root=data_dir, split=split, target_type='attr',
                              download=download)
        super(CelebALoader, self).__init__(dataset=self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           train_transform=train_transform,
                                           val_transform=val_transform,
                                           validation_split=validation_split,
                                           num_workers=num_workers,
                                           sampler=sampler,
                                           collate_fn=collate_fn,
                                           drop_last=drop_last,
                                           seed=seed)

    def __repr__(self):
        return "\n".join([self.dataset.__repr__(),
                          super(CelebALoader, self).__repr__()])

    def set_transform(self, dataset, transform):
        new_dataset = copy.deepcopy(dataset)
        new_dataset.transform = transform
        return new_dataset

    @property
    def split(self):
        return self.dataset.split
