import torch
import torch.nn as nn


class FeatureMasksGenerator(nn.Module):

    def __init__(self, size, coverage):
        self.size = size
        self.min_coverage = coverage[0]
        self.max_coverage = coverage[1]

    def random_coverage(self):
        p = (self.max_coverage - self.min_coverage) * torch.rand(1).item() + self.min_coverage
        return p

    def forward(self, batch_size=1):
        # Generate available features mask
        p = self.random_coverage()
        a = torch.rand(self.size) > p

        # Generate required features mask
        p = self.random_coverage()
        r = torch.rand(self.size) > p

        # Zero out common features in required features
        r = (r.float() - a.float()) > 0

        # Tile mask at batch size
        a = a.unsqueeze(0).repeat((batch_size, 1, 1))
        r = r.unsqueeze(0).repeat((batch_size, 1, 1))
        return a, r
