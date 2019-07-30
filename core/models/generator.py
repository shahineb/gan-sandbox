import torch
import torch.nn as nn
from .backbones import ConvNet
from .modules import ConvTranspose2d


class Generator(ConvNet):
    """Basic convolutional discriminator network

    Args:
        latent_size (tuple[int]): (C, W, H) of latent dimension
        nb_filters (list[int]): number of filters of each block
        conv_kwargs (dict, list[dict]): convolutional block kwargs, if dict same everywhere
    """
    BASE_KWARGS = {'kernel_size': 3, 'stride': 2, 'relu': True, 'bn': True}

    def __init__(self, latent_size, nb_filters, conv_kwargs=BASE_KWARGS):
        super(Generator, self).__init__(latent_size)

        # Setup network's dimensions
        C, H, W = latent_size
        self.nb_filters = nb_filters
        self.conv_kwargs = self._init_kwargs_path(conv_kwargs, nb_filters)

        # Build convolutional layers
        hidden_seq = []
        hidden_seq += [ConvTranspose2d(in_channels=C, out_channels=self.nb_filters[0], **self.conv_kwargs[0])]
        hidden_seq += [ConvTranspose2d(in_channels=self.nb_filters[i - 1], out_channels=self.nb_filters[i],
                       **self.conv_kwargs[i]) for i in range(1, len(self.nb_filters))]
        self.hidden_layers = nn.Sequential(*hidden_seq)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, C, W, H)
        """
        h = self.hidden_layers(x)
        return torch.tanh(h)

    @property
    def latent_size(self):
        return self.input_size
