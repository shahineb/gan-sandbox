import copy
import random
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import io
import hashlib
from PIL import Image, ImageDraw
from nltk.corpus import words
from ._base import SelfSupervisedLoader

"""
Identicon dataset and dataloader derived from https://github.com/flavono123/identicon
identicon generator
"""


class IdenticonSet(Dataset):

    def __init__(self, size, transform=None, target_transform=None):
        self.size = size
        self.seed_words = random.choices(words.words(), k=size)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        identicon = render(self.seed_words[idx])
        img = Image.open(io.BytesIO(identicon)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class IdenticonLoader(SelfSupervisedLoader):
    """Self-supervised dataloader for identicon dataset

    Args:
        batch_size (int): how many samples per batch to load (default: 1)
        train_transform (callable): transformation for train set PIL image
        val_transform (callable): transformation for validation set PIL image
        shuffle (bool):
        validation_split (int, float): number of validation samples of fraction in ]0, 1[
        num_workers (int)
        collate_fn (callable): merges a list of samples to form a mini-batch.
    """
    def __init__(self, size, batch_size=1, train_transform=None,
                 val_transform=None, shuffle=True, validation_split=0.,
                 num_workers=1, collate_fn=default_collate,
                 drop_last=True, sampler=None, seed=SelfSupervisedLoader.SEED):
        self.batch_size = batch_size
        self.dataset = IdenticonSet(size=size, transform=train_transform)
        super(IdenticonLoader, self).__init__(dataset=self.dataset,
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
                          super(IdenticonLoader, self).__repr__()])

    def set_transform(self, dataset, transform):
        new_dataset = copy.deepcopy(dataset)
        new_dataset.transform = transform
        return new_dataset


################################################################################
BACKGROUND_COLOR = (244, 244, 244)


def render(code):
    hex_list = _to_hash_hex_list(code)
    color = _extract_color(hex_list)
    grid = _build_grid(hex_list)
    flatten_grid = _flat_to_list(grid)
    pixels = _set_pixels(flatten_grid)
    identicon_im = _draw_identicon(color, flatten_grid, pixels)

    identicon_byte_arr = io.BytesIO()
    identicon_im.save(identicon_byte_arr, format='PNG')
    identicon_byte_arr = identicon_byte_arr.getvalue()

    return identicon_byte_arr


def _to_hash_hex_list(code):
    # TODO: Choose hash scheme
    hash = hashlib.md5(code.encode('utf8'))
    return hash.hexdigest()


def _extract_color(hex_list):
    r, g, b = tuple(hex_list[i:i + 2] for i in range(0, 2 * 3, 2))
    return f'#{r}{g}{b}'


def _set_pixels(flatten_grid):
    # len(list) should be a squared of integer value
    # Caculate pixels
    pixels = []
    for i, val in enumerate(flatten_grid):
        x = int(i % 5 * 50) + 20
        y = int(i // 5 * 50) + 20

        top_left = (x, y)
        bottom_right = (x + 50, y + 50)

        pixels.append([top_left, bottom_right])

    return pixels


def _build_grid(hex_list):
    # Tailing hex_list to rear 15 bytes
    hex_list_tail = hex_list[2:]

    # Make 3x5 gird, half of the symmetric grid(left side)
    hex_half_grid = [[hex_list_tail[col:col + 2] for col in range(row, row + 2 * 3, 2)]
                     for row in range(0, 2 * 3 * 5, 2 * 3)]

    hex_grid = _mirror_row(hex_half_grid)

    int_grid = [list(map(lambda e: int(e, base=16), row)) for row in hex_grid]

    # TODO: Using more entropies, should be deprecated
    filtered_grid = [[byte if byte % 2 == 0 else 0 for byte in row] for row in int_grid]

    return filtered_grid


def _mirror_row(half_grid):
    opposite_half_grid = [list(reversed(row)) for row in half_grid]
    # FIXME: just for odd(5) num column now
    grid = [row + mirrored_row[1:] for row, mirrored_row in zip(half_grid, opposite_half_grid)]
    return grid


def _flat_to_list(nested_list):
    flatten_list = [e for row in nested_list for e in row]
    return flatten_list


def _draw_identicon(color, grid_list, pixels):
    identicon_im = Image.new('RGB', (50 * 5 + 20 * 2, 50 * 5 + 20 * 2), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(identicon_im)
    for grid, pixel in zip(grid_list, pixels):
        if grid != 0:  # for not zero
            draw.rectangle(pixel, fill=color)
    identicon_im = _crop_coner_round(identicon_im, 50)
    return identicon_im


def _crop_coner_round(im, rad):
    round_edge = Image.new('L', (rad * 2, rad * 2), 0)

    draw = ImageDraw.Draw(round_edge)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill='white')

    alpha = Image.new('L', im.size, 255)

    w, h = im.size

    alpha.paste(round_edge.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(round_edge.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(round_edge.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(round_edge.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))

    im.putalpha(alpha)
    return im
