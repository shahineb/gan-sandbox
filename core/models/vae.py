import numpy as np
import torch
import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d, ConvTranspose2d


class VAE(ConvNet):
    """Variational AutoEncoder implementation

    Args:
        input_size (tuple): (C, W, H)
        h_dim (int): hidden space dimensionality
        z_dim (int): latent space dimensionality
        enc_nf (list[int]): number of filters of encoding path
        dec_nf (list[int]): number of filters of decoding path
        enc_kwargs (dict, list[dict]): kwargs of encoding path, if dict same everywhere
        dec_kwargs (dict, list[dict]): kwargs of decoding path, if dict same everywhere
        out_kwargs (dict): kwargs of output layer
    """
    BASE_KWARGS = {'kernel_size': 3, 'stride': 2, 'relu': True}

    def __init__(self, input_size, z_dim, enc_nf, dec_nf, enc_kwargs=BASE_KWARGS,
                 dec_kwargs=BASE_KWARGS, out_kwargs=BASE_KWARGS):
        super(VAE, self).__init__(input_size)

        # Setup network's dimensions
        C, W, H = input_size
        self.z_dim = z_dim
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        self.enc_kwargs = VAE._init_kwargs_path(enc_kwargs, enc_nf)
        self.dec_kwargs = VAE._init_kwargs_path(dec_kwargs, dec_nf)
        self.out_kwargs = out_kwargs

        # Build encoding path
        encoding_seq = [Conv2d(in_channels=C, out_channels=self.enc_nf[0], **self.enc_kwargs[0])]
        encoding_seq += [Conv2d(in_channels=self.enc_nf[i - 1], out_channels=self.enc_nf[i],
                         **self.enc_kwargs[i]) for i in range(1, len(self.enc_nf))]
        self.encoder = nn.Sequential(*encoding_seq)
        self.h_dim = self._hidden_dimension_numel()

        # Build bottleneck layers
        try:
            self.fc1 = nn.Linear(self.h_dim, z_dim)
            self.fc2 = nn.Linear(self.h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, self.h_dim)
        except ZeroDivisionError:
            raise ZeroDivisionError("Overpooled input")

        # Build decoding layers
        decoding_seq = [ConvTranspose2d(self.h_dim, out_channels=self.dec_nf[0], **self.dec_kwargs[0])]
        decoding_seq += [ConvTranspose2d(in_channels=self.dec_nf[i - 1], out_channels=self.dec_nf[i],
                         **self.dec_kwargs[i]) for i in range(1, len(self.dec_nf))]
        decoding_seq += [ConvTranspose2d(self.dec_nf[-1], C, **self.out_kwargs)]
        self.decoder = nn.Sequential(*decoding_seq)

    def _hidden_dimension_numel(self):
        """Computes number of elements of hidden dimension
        """
        image_size = self.input_size
        for conv_block in self.encoder:
            image_size = conv_block.output_size(image_size)
        return np.prod(image_size)

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick (Kingma 2014)
        Args:
            mu (torch.Tensor)
            logvar (torch.Tensor)
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size())
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        """Bottleneck layer
        Args:
            h (torch.Tensor): (N, h_dim)
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, C, W_in, H_in)

        Returns:
            (torch.Tensor): (N, C, W_out, H_out)

        """
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.view(h.size(0), -1))
        z = self.fc3(z)
        output = self.decoder(z.view(-1, self.h_dim, 1, 1))
        return output
