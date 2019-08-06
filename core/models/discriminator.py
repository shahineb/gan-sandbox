import numpy as np
import torch
import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d


class Discriminator(ConvNet):
    """Basic convolutional discriminator network

    Args:
        input_size (tuple): (C, W, H)
        nb_filters (list[int]): number of filters of each block
        conv_kwargs (dict, list[dict]): convolutional block kwargs, if dict same everywhere
        output_dim (int): output dimensionality
    """
    BASE_KWARGS = {'kernel_size': 3, 'stride': 2, 'bias': False, 'relu': True, 'bn': True}

    def __init__(self, input_size, nb_filters, conv_kwargs=BASE_KWARGS, output_dim=1):
        super(Discriminator, self).__init__(input_size)

        # Setup network's dimensions
        C, H, W = input_size
        self.nb_filters = nb_filters
        self.conv_kwargs = self._init_kwargs_path(conv_kwargs, nb_filters)
        self.output_dim = output_dim

        # Build convolutional layers
        hidden_seq = []
        hidden_seq += [Conv2d(in_channels=C, out_channels=self.nb_filters[0], **self.conv_kwargs[0])]
        hidden_seq += [Conv2d(in_channels=self.nb_filters[i - 1], out_channels=self.nb_filters[i],
                       **self.conv_kwargs[i]) for i in range(1, len(self.nb_filters))]

        self.hidden_layers = nn.Sequential(*hidden_seq)
        self.h_dim = self._hidden_dimension_numel()

        # Build and initialize output layer
        try:
            self.output_layer = Conv2d(in_channels=self.nb_filters[-1], out_channels=output_dim,
                                       kernel_size=int(input_size[1] / 16), stride=1, relu=False, bn=False)
        except ZeroDivisionError:
            raise ZeroDivisionError("Overpooled input")

    def _hidden_dimension_numel(self):
        """Computes number of elements of hidden dimension
        """
        image_size = self.input_size
        for conv_block in self.hidden_layers:
            image_size = conv_block.output_size(image_size)
        return np.prod(image_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, C, W, H)
        """
        h = self.hidden_layers(x)
        output = self.output_layer(h)
        return output.squeeze()


class CDiscriminator(Discriminator):

    BASE_KWARGS = {'kernel_size': 3, 'stride': 2, 'bias': False, 'relu': True, 'bn': True}

    def __init__(self, input_size, nb_class, nb_filters, conv_kwargs=BASE_KWARGS, output_dim=1):
        # Setup class embedding layer
        C, H, W = input_size
        self.embedding = nn.Sequential([nn.Embedding(nb_class, nb_class),
                                        nn.Linear(nb_class, H * W)])

        super(CDiscriminator, self).__init__(latent_size=(C + 1, H, W),
                                             nb_filters=nb_filters,
                                             conv_kwargs=conv_kwargs)

    def forward(self, x, labels):
        y = self.embedding(labels).view_as(x)
        inputs = torch.cat([x, y], dim=1)
        return super(CDiscriminator, self).forward(inputs)
