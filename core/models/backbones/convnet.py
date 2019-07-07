from abc import abstractmethod
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, input_size):
        """General class describing networks with convolutional layers

        Args:
            input_size (tuple): (C, H, W)
        """
        super(ConvNet, self).__init__()
        self._input_size = input_size

    @staticmethod
    def _init_kwargs_path(conv_kwargs, nb_filters):
        """Initializes encoding or decoding path making sure making sure it
        matches the number of filters dimensions

        Args:
            conv_kwargs (dict, list[dict]): convolutional block kwargs
            nb_filters (list[int]): number of filter of each block
        """
        if isinstance(conv_kwargs, list):
            assert len(conv_kwargs) == len(nb_filters), "Kwargs and number of filters length must match"
            return conv_kwargs
        elif isinstance(conv_kwargs, dict):
            return len(nb_filters) * [conv_kwargs]
        else:
            raise TypeError("kwargs must be of type dict or list[dict]")

    def _hidden_dimension_numel(self):
        """Computes number of elements of hidden dimension
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, C, W, H)
        """
        pass

    @property
    def input_size(self):
        return self._input_size
