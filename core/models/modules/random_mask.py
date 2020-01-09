import numpy as np
import torch
import torch.nn as nn


class FeatureMasksGenerator(nn.Module):
    """Class to generate image masks as in Belghazi et. al 2019 -
        Learning about an exponential amount of conditional distributions

    Args:
        size (tuple[int]): (width, height)
        coverage (tuple[float]): (min_coverage, max_coverage) percentage of
            pixels allowed to be masked
    """

    def __init__(self, size, coverage):
        super(FeatureMasksGenerator, self).__init__()
        self.size = size
        self.min_coverage = coverage[0]
        self.max_coverage = coverage[1]

    def rectangular_mask(self):
        width, height = self.size
        # Compute random ratios s.t min_coverage * w * h <= ratio * w * h <= max_coverage * w * h
        width_ratio = np.sqrt(self.min_coverage) + (np.sqrt(self.max_coverage) - np.sqrt(self.min_coverage)) * np.random.rand()
        height_ratio = np.sqrt(self.min_coverage) + (np.sqrt(self.max_coverage) - np.sqrt(self.min_coverage)) * np.random.rand()

        # Derive mask dimensions and origin
        mask_width = int(np.floor(width * width_ratio))
        mask_height = int(np.floor(height * height_ratio))
        x0 = int(np.floor((width - mask_width) * np.random.rand()))
        y0 = int(np.floor((height - mask_height) * np.random.rand()))

        # Build mask as (width, height) tensor
        mask = torch.ones(self.size)
        mask[x0:x0 + mask_width, y0:y0 + mask_height] = 0.
        return mask

    def forward(self, batch_size=1):
        # Generate available and required features masks
        a = self.rectangular_mask()
        r = self.rectangular_mask()

        # Zero out common features in required features
        r = (r.float() - a.float()) > 0

        # Tile mask at batch size
        a = a.unsqueeze(0).repeat((batch_size, 1, 1)).float()
        r = r.unsqueeze(0).repeat((batch_size, 1, 1)).float()
        return a, r
