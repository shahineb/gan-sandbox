import os
import copy
from torch.utils.data import Dataset
import utils.IOHandler as io
from PIL import Image
from ._base import SelfSupervisedLoader
from .utils import collate


class PokemonDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.idx2key = io.load_json(os.path.join(root, "index.json"))
        self.types = dict()
        self.root_dir = root
        for scrapset in ["artworks", "pokebip"]:
            self.types.update(**io.load_json(os.path.join(root, scrapset, "types.json")))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.types)

    def __getitem__(self, idx):
        file_name = self.idx2key[str(idx)]
        img_path = os.path.join(self.root_dir, file_name)
        img = Image.open(img_path).convert('RGB')
        label = self.types[file_name]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class PokeLoader(SelfSupervisedLoader):
    """Self-supervised dataloader for pokemon dataset

    Args:
        data_dir (str): dataset root directory
        batch_size (int): how many samples per batch to load (default: 1)
        train_transform (callable): transformation for train set PIL image
        val_transform (callable): transformation for validation set PIL image
        shuffle (bool):
        validation_split (int, float): number of validation samples of fraction in ]0, 1[
        num_workers (int)
        collate_fn (callable): merges a list of samples to form a mini-batch.
    """
    def __init__(self, data_dir, batch_size=1, train_transform=None,
                 val_transform=None, shuffle=True, validation_split=0.,
                 num_workers=1, collate_fn=collate.drop_target,
                 drop_last=True, sampler=None, seed=SelfSupervisedLoader.SEED):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = PokemonDataset(root=data_dir, transform=train_transform)
        super(PokeLoader, self).__init__(dataset=self.dataset,
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
                          super(PokeLoader, self).__repr__()])

    def set_transform(self, dataset, transform):
        new_dataset = copy.deepcopy(dataset)
        new_dataset.transform = transform
        return new_dataset
