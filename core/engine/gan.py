import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from progress.bar import Bar
from .trainer import Trainer

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from core.models import Discriminator
from core.models import utils


class GANTrainer(Trainer):

    def __init__(self, config,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 verbose=1, chkpt=True, tensorboard=True, multigpu=True):
        super(GANTrainer, self).__init__(config=config, device=device,
                                         verbose=verbose, chkpt=chkpt,
                                         tensorboard=tensorboard,
                                         multigpu=multigpu)

        # Setup discriminator
        self.discriminator = Discriminator(**self.config.kwargs["discriminator"]).to(self.device)
        self.disc_optimizer = torch.optim.Adam(params=self.discriminator.parameters(),
                                               **self.config.kwargs["disc_optimizer"])
        # self.config.set_lr_scheduler(optim.lr_scheduler.ExponentialLR(optimizer=self.disc_optimizer, gamma=0.999))

        # init weights
        self.model.apply(utils.weight_init)
        self.discriminator.apply(utils.weight_init)

    def _compute_loss(self, output_real_sample, output_fake_sample):
        """Adversarial networks loss computation given by :

            LossDisc = E_{x~realdata}[-logD(x)] + E_{z~inputs}[-log(1 - D(G(z)))]
            LossGen = E_{z~inputs}[-logD(z)]

            We approximate:
                E_{x~realdata}[-logD(x)] = Avg(CrossEnt_{x:realbatch}(1, D(x)))
                E_{z~inputs}[-log(1 - D(G(z)))] = Avg(CrossEnt_{x:fakebatch}(0, D(x)))
                E_{z~inputs}[-logD(z)] = Avg(CrossEnt_{x:fakebatch}(1, D(x)))
        Args:
            output_real_sample (torch.Tensor): (N, ) discriminator pred on real samples
            output_fake_sample (torch.Tensor): (N, ) discriminator pred on fake samples
        """
        # Setup targets vectors
        target_real_sample = torch.ones_like(output_real_sample)
        target_fake_sample = torch.zeros_like(output_fake_sample)

        # Losses computation, criterion should be crossentropy
        loss_real_sample = self.criterion(output_real_sample, target_real_sample)
        loss_fake_sample = self.criterion(output_fake_sample, target_fake_sample)
        disc_loss = loss_real_sample + loss_fake_sample
        gen_loss = self.criterion(output_fake_sample, target_real_sample)
        return gen_loss, disc_loss

    def _eval_metrics(self, output_real_sample, output_fake_sample):
        # Setup complete outputs and targets vectors
        target_real_sample = torch.ones_like(output_real_sample)
        target_fake_sample = torch.zeros_like(output_fake_sample)
        output = torch.cat([output_real_sample, output_fake_sample])
        target = torch.cat([target_real_sample, target_fake_sample])

        # Compute generator and discriminator metrics
        fooling_rate = self.metrics[0](output_fake_sample, target_real_sample)
        precision = self.metrics[1](output, target)
        recall = self.metrics[2](output, target)
        return np.array([fooling_rate, precision, recall])

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

        # Update weights
        self.disc_optimizer.step()

        return output_real_sample, loss_real_sample, loss_fake_sample

    def _step_generator(self, latent_sample):
        # Zero out any remaining gradient
        self.model.zero_grad()
        self.discriminator.zero_grad()
        self.optimizer.zero_grad()

        # Forward pass on fake data data
        fake_sample = self.model(latent_sample)
        output_fake_sample = torch.sigmoid(self.discriminator(fake_sample))

        # Loss + backward on real sample batch
        target_real_sample = torch.ones_like(output_fake_sample)
        gen_loss = self.criterion(output_fake_sample, target_real_sample)
        gen_loss.backward()

        self.optimizer.step()

        return output_fake_sample, gen_loss

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
            # Move samples to device
            data = data.to(self.device)

            # Generate random input from latent space N(0, I) and train discriminator
            z = torch.randn((dataloader.batch_size, ) + self.model.latent_size, device=self.device)
            output_real_sample, loss_real_sample, loss_fake_sample = self._step_discriminator(data, z)

            # Generate another random latent variable and train generator
            z = torch.randn((dataloader.batch_size, ) + self.model.latent_size, device=self.device)
            output_fake_sample, gen_loss = self._step_generator(z)

            # Record loss values
            disc_loss = loss_real_sample + loss_fake_sample
            total_disc_loss += disc_loss.item()
            total_gen_loss += gen_loss.item()

            # run metrics computation on training data
            if self.metrics:
                total_metrics += self._eval_metrics(output_real_sample,
                                                    output_fake_sample)

            if batch_idx % self._log_steps == 0:
                bar.suffix = "{}/{} ({:.0f}%%) | GenLoss: {:.4f} | DiscLoss {:.4f} | FoolingRate {:.3f}| Precision {:.3f}| Recall {:.3f}".format(
                             batch_idx * dataloader.batch_size,
                             dataloader.n_train_samples,
                             100.0 * batch_idx / len(dataloader),
                             total_gen_loss / (batch_idx + 1),
                             total_disc_loss / (batch_idx + 1),
                             total_metrics[0] / (batch_idx + 1),
                             total_metrics[1] / (batch_idx + 1),
                             total_metrics[2] / (batch_idx + 1))
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

        Args:d
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
            for batch_idx, data in enumerate(dataloader):
                # Generate random input from latent space N(0, I)
                z = torch.randn((dataloader.batch_size,) + self.model.latent_size)

                # Move inputs to device
                real_sample, z = data.to(self.device), z.to(self.device)

                # Forward pass on generator
                fake_sample = self.model(z)

                # Forward pass on discriminator
                output_real_sample = torch.sigmoid(self.discriminator(real_sample))
                output_fake_sample = torch.sigmoid(self.discriminator(fake_sample))

                # Compute loss
                gen_loss, disc_loss = self._compute_loss(output_real_sample,
                                                         output_fake_sample)

                # Record loss values
                total_disc_loss += disc_loss.item()
                total_gen_loss += gen_loss.item()

                # update epoch loss and metrics
                if self.metrics:
                    total_metrics += self._eval_metrics(output_real_sample,
                                                        output_fake_sample)

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
        # Generate random input from latent space N(0, I)
        torch.random.manual_seed(self.seed)
        z = torch.randn((len(data),) + self.model.latent_size).to(self.device)

        # Forward pass on neural conditioner
        with torch.no_grad():
            fake_sample = 0.5 * (self.model(z) + 1)

        self.writer.add_image(tag='generated_samples',
                              img_tensor=make_grid(fake_sample.cpu(), nrow=8, normalize=True),
                              global_step=epoch)
        self.writer.add_image(tag='real_samples',
                              img_tensor=make_grid(torch.stack(data).cpu(), nrow=8, normalize=True),
                              global_step=epoch)

    def _histogram_callback(self, epoch):
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto', global_step=epoch)
        for name, p in self.discriminator.named_parameters():
            self.writer.add_histogram(name, p, bins='auto', global_step=epoch)
