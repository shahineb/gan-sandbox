import os
import sys
import torch
from .gan import GANTrainer

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from core.models.modules import Mixup


class MixGANTrainer(GANTrainer):

    def __init__(self, config,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 verbose=1, chkpt=True, tensorboard=True, multigpu=True):
        super(MixGANTrainer, self).__init__(config, device, verbose, chkpt,
                                            tensorboard, multigpu)
        self.mixup = Mixup(**self.config.kwargs["mixup"])

    def _step_discriminator(self, real_sample, latent_sample):
        # Zero any remaining gradient
        self.discriminator.zero_grad()
        self.disc_optimizer.zero_grad()

        # Forward pass on real data
        output_real_sample = torch.sigmoid(self.discriminator(real_sample))

        # Loss + backward on real sample batch with label smoothing
        target_real_sample = 0.9 + 0.1 * torch.rand_like(output_real_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)
        loss_real_sample.backward()

        # Generate fake sample batch + forward pass, note we detach fake samples to not backprop though generator
        fake_sample = self.model(latent_sample)
        output_fake_sample = torch.sigmoid(self.discriminator(fake_sample.detach()))

        # Loss + backward on fake batch
        target_fake_sample = torch.zeros_like(output_fake_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)
        loss_fake_sample.backward()

        # mixup step
        do_mix = torch.rand(1).item() > 0.5
        if do_mix:
            target_fake_sample = 0.1 - 0.1 * torch.rand_like(target_fake_sample)
            mixed_samples, mixed_targets = self.mixup(real_sample, target_real_sample, fake_sample, target_fake_sample)
            output_mixed_samples = torch.sigmoid(self.discriminator(mixed_samples))
            loss_mixed_samples = self.criterion(output_mixed_samples, mixed_targets)
            loss_mixed_samples.backward()

        # Update weights
        self.disc_optimizer.step()

        return output_real_sample, loss_real_sample, loss_fake_sample
