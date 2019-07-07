import numpy as np
import torch.nn as nn


class Conv2d(nn.Module):
    """Conv2d + BatchNorm + ReLU

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        relu (bool): if True, uses ReLU
        bn (bool): if True, uses batch normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, relu=False, bn=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

    def output_size(self, input_size):
        """Computes output size

        Args:
            input_size (tuple): (C_in, H_in, W_in)
        """
        _, H_in, W_in = input_size
        C_out = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        padding = self.conv.padding[0]
        stride = self.conv.stride[0]
        H_out = int(np.floor((H_in - kernel_size + 2 * padding) / stride + 1))
        W_out = int(np.floor((W_in - kernel_size + 2 * padding) / stride + 1))
        return (C_out, H_out, W_out)


class ConvTranspose2d(nn.Module):
    """Conv2d + BatchNorm + ReLU

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): stride for the cross-correlation. Default: 1
        output_padding (int or tuple, optional): controls the additional size added to one side of the output shape. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        relu (bool): if True, uses ReLU
        bn (bool): if True, uses batch normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 output_padding=0, dilation=1, relu=False, bn=False):
        super(ConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       output_padding=output_padding,
                                       dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

    def output_size(self, input_size):
        """Computes output size

        Args:
            input_size (tuple): (C, H_in, W_in)
        """
        raise NotImplementedError
