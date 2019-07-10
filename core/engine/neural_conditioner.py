import os
import sys
import numpy as np
import torch
from torchvision.utils import make_grid
from progress.bar import Bar
from .trainer import Trainer

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from core.models import modules, Discriminator


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
        self.discriminator = Discriminator(**self.config.kwargs["discriminator"]).to(self.device)
        self.disc_optimizer = torch.optim.Adam(params=self.discriminator.parameters(),
                                               **self.config.kwargs["disc_optimizer"])

    def _compute_loss(self, output_real_sample, output_fake_sample):
        target_real_sample = torch.ones_like(output_real_sample)
        target_fake_sample = torch.zeros_like(output_fake_sample)
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)
        disc_loss = loss_real_sample + loss_fake_sample
        gen_loss = self.criterion(output_fake_sample, target_real_sample)
        return gen_loss, disc_loss

    def _train_epoch(self, epoch, dataloader):
        """Run training on an epoch.

        Args:
            epoch (int): epoch number
            dataloader (core.dataloader._base.BaseDataLoader)
        """
        # prepare model for training
        self.model.train()
        self.discriminator.train()

        # init  progress bar, losses counters and metrics
        bar = Bar(f'Epoch {epoch + 1}', max=len(dataloader))
        total_disc_loss = 0
        total_gen_loss = 0
        if self.metrics:
            total_metrics = np.zeros(len(self.metrics))

        for batch_idx, data in enumerate(dataloader):

            # Generate available and requested features masks and noise tensor
            a, r = self.mask_generator(batch_size=dataloader.batch_size)
            z = torch.rand((dataloader.batch_size,) + data.shape[-2:]).unsqueeze(1)

            # Move inputs to device
            data = data.to(self.device)
            a, r, z = a.to(self.device), r.to(self.device), z.to(self.device)

            # Build vae input and real sample
            a_, r_ = a.unsqueeze(1), r.unsqueeze(1)
            inputs = torch.cat([data.mul(a_), a_, r_, z], dim=1)
            real_sample = data.mul(r_)

            # Forward pass on neural conditioner
            fake_sample = self.model(inputs)

            # Forward pass on discriminator
            output_real_sample = self.discriminator(real_sample)
            output_fake_sample = self.discriminator(fake_sample)

            # Compute loss
            gen_loss, disc_loss = self._compute_loss(torch.sigmoid(output_real_sample),
                                                     torch.sigmoid(output_fake_sample))

            # Backward + optimize
            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()
            gen_loss.backward(retain_graph=True)
            disc_loss.backward()
            self.optimizer.step()
            self.disc_optimizer.step()

            # Record loss values
            total_disc_loss += disc_loss.item()
            total_gen_loss += gen_loss.item()

            # run metrics computation on training data
            if self.metrics:
                total_metrics += self._eval_metrics(inputs, torch.ones(data.size(0)))

            if batch_idx % self._log_steps == 0:
                bar.suffix = "{}/{} ({:.0f}%%) | GenLoss: {:.6f} | DiscLoss {:.6f}".format(
                             batch_idx * dataloader.batch_size,
                             dataloader.n_train_samples,
                             100.0 * batch_idx / len(dataloader),
                             total_gen_loss / (batch_idx + 1),
                             total_disc_loss / (batch_idx + 1))
                bar.next(self._log_steps)

        # sum up dictionnary
        logs = {'gen_loss': total_gen_loss / len(dataloader),
                'disc_loss': total_disc_loss / len(dataloader)}
        if self.metrics:
            total_metrics = total_metrics / len(dataloader)
            logs.update({"train/" + metric.__name__: total_metrics[i] for i, metric in enumerate(self.metrics)})
        return logs

    def _valid_epoch(self, epoch, dataloader):
        """Run validation on an epoch

        Args:
            epoch (int): epoch number
            dataloader (core.dataloader._base.BaseDataLoader)
        """
        # prepare model for inference
        self.model.eval()
        self.discriminator.eval()

        total_disc_loss = 0
        total_gen_loss = 0
        # init epoch metrics
        if self.metrics:
            total_metrics = np.zeros(len(self.metrics))

        # run validation loop
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                # Generate available and requested features masks and noise tensor
                a, r = self.mask_generator(batch_size=dataloader.batch_size)
                z = torch.rand((dataloader.batch_size,) + data.shape[-2:]).unsqueeze(1)

                # Move inputs to device
                data = data.to(self.device)
                a, r, z = a.to(self.device), r.to(self.device), z.to(self.device)

                # Build vae input and real sample
                a_, r_ = a.unsqueeze(1), r.unsqueeze(1)
                inputs = torch.cat([data.mul(a_), a_, r_, z], dim=1)
                real_sample = data.mul(r_)

                # Forward pass on neural conditioner
                fake_sample = self.model(inputs)

                # Forward pass on discriminator
                output_real_sample = self.discriminator(real_sample)
                output_fake_sample = self.discriminator(fake_sample)

                # Compute loss
                gen_loss, disc_loss = self._compute_loss(output_real_sample, output_fake_sample)

                # Record loss values
                total_disc_loss += disc_loss.item()
                total_gen_loss += gen_loss.item()

                # update epoch loss and metrics
                if self.metrics:
                    total_metrics += self._eval_metrics(data, target)

        # sum up dictionnary
        logs = {'gen_loss': total_gen_loss / len(dataloader),
                'disc_loss': total_disc_loss / len(dataloader)}

        if self.metrics:
            total_metrics = total_metrics / len(dataloader)
            logs.update({"val/" + metric.__name__: total_metrics[i] for i, metric in enumerate(self.metrics)})
        return logs

    def _image_callback(self, data, target, epoch):
        """Image dumping callback for tensorboard

        Args:
            data (torch.Tensor): images batch
            epoch (int): epoch number
        """
        # Generate available and requested features masks and noise tensor
        a, r = self.mask_generator(batch_size=data.size(0))
        z = torch.rand((data.size(0),) + data.shape[-2:]).unsqueeze(1)

        # Move inputs to device
        data = data.to(self.device)
        a, r, z = a.to(self.device), r.to(self.device), z.to(self.device)

        # Build vae input
        a_, r_ = a.unsqueeze(1), r.unsqueeze(1)
        inputs = torch.cat([data.mul(a_), a_, r_, z], dim=1)

        # Forward pass on neural conditioner
        with torch.no_grad():
            fake_sample = self.model(inputs)

        self.writer.add_image(tag='conditioned input',
                              img_tensor=make_grid(data.mul(a_).cpu(), nrow=8, normalize=True),
                              global_step=epoch)
        self.writer.add_image(tag='generated samples',
                              img_tensor=make_grid(fake_sample.cpu(), nrow=8, normalize=True),
                              global_step=epoch)

    def _histogram_callback(self, epoch):
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto', global_step=epoch)
        for name, p in self.discriminator.named_parameters():
            self.writer.add_histogram(name, p, bins='auto', global_step=epoch)
