import torch
import torch.nn as nn
import numpy as np


class Mixup(nn.Module):
    """Zhang et al. 2018

    Args:
        alpha (float): parameter for beta distribution
        fav_src (bool): if True, favors first term in interpolation as proposed
        by Berthelot et. al 2019
    """

    def __init__(self, alpha=0.75, fav_src=False):
        super(Mixup, self).__init__()
        assert alpha > 0, "Must specifiy positive value for alpha"
        self._alpha = alpha
        self._fav_src = fav_src

    def mix_pair(self, x0, l0, x1, l1):
        """Mixes pair of sample-label batches
        Args:
            x0 (torch.Tensor)
            l0 (torch.Tensor)
            x1 (torch.Tensor)
            l1 (torch.Tensor)
        """
        lbda = np.random.beta(self.alpha, self.alpha)
        if self.fav_src:
            lbda = np.maximum(lbda, 1 - lbda)
        index = torch.randperm(x0.size()[0])
        mixed_sample = lbda * x0 + (1 - lbda) * x1[index]
        mixed_targets = lbda * l0 + (1 - lbda) * l1[index]
        return mixed_sample, mixed_targets

    def mix(self, x, l):
        return self.mix_pair(x, l, x, l)

    def forward(self, x0, l0, x1, l1):
        return self.mix_pair(x0, l0, x1, l1)

    @property
    def alpha(self):
        return self._alpha

    @property
    def fav_src(self):
        return self._fav_src

    def set_alpha(self, alpha):
        self._alpha = alpha
