import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from progress.bar import Bar
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
        self.disc_optimizer = self.config.kwargs["discriminator_optimizer"]  # TO BE CHECKED

    def _train_epoch(self, epoch, dataloader):
        """Run training on an epoch.

        Args:
            epoch (int): epoch number
            dataloader (core.dataloader._base.BaseDataLoader)
        """
        # prepare model for training
        self.model.train()

        # init  progress bar, losses counters and metrics
        bar = Bar(f'Epoch {epoch + 1}', max=len(dataloader))
        total_disc_loss = 0
        total_gen_loss = 0
        total_loss = 0
        if self.metrics:
            total_metrics = np.zeros(len(self.metrics))

        for batch_idx, data in enumerate(dataloader):

            # Generate available and requested features masks and noise tensor
            a, r = self.mask_generator(batch_size=self.dataloader.batch_size)
            z = torch.rand(data.size())

            # Build vae input and real sample
            a_ = a.unsqueeze(1).expand_as(data)
            r_ = r.unsqueeze(1).expand_as(data)
            inputs = torch.stack([data.mul(a_), a, r, z])
            real_sample = torch.stack([inputs.mul(a_), inputs.mul(r_), a, r])

            # Forward pass on neural conditioner
            fake_sample = self.model(inputs)

            # Forward pass on discriminator
            output_real_sample = self.discriminator(real_sample)
            output_fake_sample = self.discriminator(fake_sample)

            # Compute loss
            gen_loss, disc_loss = self._compute_loss(output_real_sample, output_fake_sample)

            # Backward + optimize
            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()
            gen_loss.backward()
            disc_loss.backward()
            self.optimizer.step()
            self.disc_optimizer.step()

            # Record loss values
            total_disc_loss += disc_loss.item()
            total_gen_loss += gen_loss.item()

            # run metrics computation on training data
            if self.metrics:
                eval_data = torch.cat([data, fake_sample])
                total_metrics += self._eval_metrics(data, torch.ones(data.size(0)))
