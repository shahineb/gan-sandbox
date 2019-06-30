import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .trainer import Trainer

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from core.models import modules


class NCTrainer(Trainer):

    def __init__(self, config,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 verbose=1, chkpt=True, tensorboard=True, multigpu=True):
        super(NCTrainer, self).__init__(config=config, device=device,
                                        verbose=verbose, chkpt=chkpt,
                                        tensorboard=tensorboard,
                                        multigpu=multigpu)
        # setup mask generation instance and discriminator
        self.mask_generator = modules.FeatureMasksGenerator(**self.config.kwargs["masks"])
        self.discriminator = modules.Discriminator(**self.configs.kwargs["discriminator"])
        self.disc_optimizer = self.config.kwargs["discriminator_optimizer"]

    def _train_epoch(self, epoch, dataloader):
        """Run training on an epoch.

        Args:
            epoch (int): epoch number
            dataloader (core.dataloader._base.BaseDataLoader)
        """
        # prepare model for training
        self.model.train()

        for batch_idx, data in enumerate(dataloader):

            # Generate available and requested features masks and noise tensor
            a, r = self.mask_generator(batch_size=self.dataloader.batch_size)
            z = torch.rand(data.size())

            # Build input and real target
            inputs = torch.stack([data * a, a, r, z])
            real_target = torch.stack([data * a, data * r, a, r])

            # Forward pass on neural conditioner
            fake_target = self.model(inputs)

            # Forward pass on discriminator
            output_real_target = self.discriminator(real_target)
            output_fake_target = self.discriminator(fake_target)

            # Compute loss
            gen_loss, disc_loss = self._compute_loss(output_real_target, output_fake_target)

            # Backward + optimize
            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()
            gen_loss.backward()
            disc_loss.backward()
            self.optimizer.step()
            self.disc_optimizer.step()
