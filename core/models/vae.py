import torch
import torch.nn as nn
from .modules import Conv2d, ConvTranspose2d


class VAE(nn.Module):
    """Variational AutoEncoder implementation

    Args:
        channels (int):
        h_dim (int): hidden space dimensionality
        z_dim (int): latent space dimensionality
        enc_nf (list[int]): number of filters of encoding path
        dec_nf (list[int]): number of filters of decoding path
        enc_kwargs (dict, list[dict]): kwargs of encoding path, if dict same everywhere
        dec_kwargs (dict, list[dict]): kwargs of decoding path, if dict same everywhere
        out_kwargs (dict): kwargs of output layer
    """
    BASE_KWARGS = {'kernel_size': 3, 'stride': 2, 'relu': True}

    def __init__(self, channels, h_dim, z_dim, enc_nf, dec_nf,
                 enc_kwargs=BASE_KWARGS, dec_kwargs=BASE_KWARGS, out_kwargs=BASE_KWARGS):
        super(VAE, self).__init__()

        # Setup network's dimensions
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        self.enc_kwargs = VAE._init_kwargs_path(enc_kwargs, enc_nf)
        self.dec_kwargs = VAE._init_kwargs_path(dec_kwargs, dec_nf)
        self.out_kwargs = out_kwargs

        # Build enconding path
        encoding_seq = [Conv2d(in_channels=channels, out_channels=self.enc_nf[0], **self.enc_kwargs[0])]
        encoding_seq += [Conv2d(in_channels=self.enc_nf[i - 1], out_channels=self.enc_nf[i],
                         **self.enc_kwargs[i]) for i in range(1, len(self.enc_nf))]
        self.encoder = nn.Sequential(*encoding_seq)

        # Build bottleneck layers
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Build decoding layers
        decoding_seq = [ConvTranspose2d(h_dim, out_channels=self.dec_nf[0], **self.dec_kwargs[0])]
        decoding_seq += [ConvTranspose2d(in_channels=self.dec_nf[i - 1], out_channels=self.dec_nf[i],
                         **self.dec_kwargs[i]) for i in range(1, len(self.dec_nf))]
        decoding_seq += [ConvTranspose2d(self.dec_nf[-1], channels, **self.out_kwargs)]
        self.decoder = nn.Sequential(*decoding_seq)

    @staticmethod
    def _init_kwargs_path(kwargs, nb_filters):
        """Initializes encoding or decoding path making sure making sure it
        matches the number of filters dimensions

        Args:
            kwargs (dict, list[dict]): enc_kwargs or dec_kwargs
            nb_filters (list[int]): enc_nf or dec_nf
        """
        if isinstance(kwargs, list):
            assert len(kwargs) == len(nb_filters), "Kwargs and number of filters length must match"
            return kwargs
        elif isinstance(kwargs, dict):
            return len(nb_filters) * [kwargs]
        else:
            raise TypeError("kwargs must be of type dict or list[dict]")

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
