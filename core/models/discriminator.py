import numpy as np
import torch.nn as nn
from .backbone import ConvNet
from .modules import Conv2d


class Discriminator(ConvNet):
    def __init__(self, input_size, nb_filters, conv_kwargs, output_dim=1):
        """Basic convolutional discriminator network

        Args:
            input_size (tuple): (C, W, H)
            nb_filters (list[int]): number of filters of each block
            conv_kwargs (dict, list[dict]): convolutional block kwargs, if dict same everywhere
            output_dim (int): output dimensionality
        """
        super(Discriminator, self).__init__(input_size)

        # Setup network's dimensions
        C, H, W = input_size
        self.nb_filters = nb_filters
        self.conv_kwargs = Discriminator._init_kwargs_path(conv_kwargs, nb_filters)
        self.h_dim = int(np.round(C * W * H / 2**len(nb_filters)))
        self.output_dim = output_dim

        # Build convolutional layers
        hidden_seq = []
        hidden_seq += [Conv2d(in_channels=C, out_channels=self.nb_filters[0], **self.conv_kwargs[0])]
        hidden_seq += [Conv2d(in_channels=self.nb_filters[i - 1], out_channels=self.nb_filters[i],
                       **self.conv_kwargs[i]) for i in range(1, len(self.nb_filters))]

        # TODO : Conv kernel intializer
        self.hidden_layers = nn.Sequential(**hidden_seq)

        # Build and initialize output layer
        self.output_layer = nn.Linear(self.h_dim, self.output_dim)
        nn.init.normal_(self.output_layer.weight, mean=0., std=0.02)
        nn.init.constant_(self.output_layerense.bias, 0.)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, C, W, H)
        """
        h = self.hidden_layer(x)
        h = h.view(-1, self.h_dim)
        out = self.output_layer(h)
        return out
