import torch
import torch.nn as nn
from .modules import Conv2d, ConvTranspose2d


class VAE(nn.Module):
    """Variational AutoEncoder implementation from
    https://github.com/sksq96/pytorch-vae/blob/master/vae.py

    Args:
        channels (int):
        h_dim (int):
        z_dim (int):
    """

    def __init__(self, channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        encoding_seq = [Conv2d(in_channels=channels, out_channels=32,
                               kernel_size=4, stride=2, relu=True)]
        encoding_seq += [Conv2d(in_channels=1 << i, out_channels=1 << (i + 1),
                                kernel_size=4, stride=2, relu=True) for i in range(5, 8)]
        self.encoder = nn.Sequential(*encoding_seq)

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        decoding_seq = [ConvTranspose2d(h_dim, 128, kernel_size=5,
                                        stride=2, relu=True)]
        decoding_seq += [ConvTranspose2d(1 << i, 1 << (i - 1), kernel_size=5,
                                         stride=2, relu=True) for i in range(7, 3, -1)]
        decoding_seq += [ConvTranspose2d(8, channels, kernel_size=5,
                                         stride=2, relu=True)]
        self.decoder = nn.Sequential(*decoding_seq)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(x)[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.view(h.size(0), -1))
        z = self.fc3(z)
        return self.decoder(z.view(-1, self.h_dim, 1, 1))
