import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, image_size, n_filters, output_dim=1):
        super(Discriminator, self).__init__()

        self.hidden_layer = nn.Sequential()
        for i in range(len(n_filters)):
            # Conv Layer
            if i == 0:
                conv = nn.Conv2d(input_dim,
                                 n_filters[i],
                                 kernel_size=5,
                                 stride=2,
                                 padding=2)
            else:
                conv = nn.Conv2d(n_filters[i - 1],
                                 n_filters[i],
                                 kernel_size=5,
                                 stride=2,
                                 padding=2)
            conv_name = "conv_" + str(i + 1)
            self.hidden_layer.add_module(conv_name, conv)

            # Conv kernel intializer
            nn.init.normal_(conv.weight, mean=0., std=0.02)
            nn.init.constant_(conv.bias, 0.)

            # Batch norm
            bn_name = "batchnorm_" + str(i + 1)
            bn = nn.BatchNorm2d(n_filters[i], affine=True)
            self.hidden_layer.add_module(bn_name, bn)

            # Activation
            act_name = "lrelu_" + str(i + 1)
            act = nn.LeakyReLU(0.2)
            self.hidden_layer.add_module(act_name, act)

        # Output Layer
        self.output_layer = nn.Sequential()
        new_image_size = int(np.round(image_size / (2**len(n_filters))))
        dense = nn.Linear(n_filters[-1] * new_image_size**2, output_dim)
        sigmoid = nn.Sigmoid()
        self.output_layer.add_module("fc", dense)
        self.output_layer.add_module("sigmoid", sigmoid)
        nn.init.normal_(dense.weight, mean=0., std=0.02)
        nn.init.constant_(dense.bias, 0.)

    def forward(self, x):
        h = self.hidden_layer(x)
        (_, C, H, W) = h.data.size()
        h = h.view(-1, C * H * W)
        out = self.output_layer(h)
        return out
