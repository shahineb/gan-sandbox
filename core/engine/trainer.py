import os
import sys
import warnings
from abc import abstractmethod
import logging
import verboselogs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .config_file import ConfigFile
from utils.decorators import validation

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from utils.decorators import serializable


@serializable
class Trainer:
    """Training module dedicated to higher level specifications than ConfigFile
    that do not impact the final outcome

    Attributes:
        _device (torch.device): {'cuda', 'cpu'}
        _config (core.engine.config_file.ConfigFile)
        _verbose (int): {0, 1}
        _chkpt (bool): if True, saves checkpoint file at the end epochs
        _tensorboard (bool): if True, dumps tensorboard logs
        _logger (verboselogs.VerboseLogger): logging instance
        _writer (torch.utils.tensorboard.SummaryWriter): if _tensorboard, writing
            instance to handle logs stream
    """

    def __init__(self, config, device, verbose, chkpt, tensorboard, multigpu):
        super(Trainer, self).__init__()
        self._device = device
        self._config = config
        self._config.model.to(self._device)
        self._verbose = verbose
        self._logger = verboselogs.VerboseLogger('demo')
        self._logger.addHandler(logging.StreamHandler())
        self._logger.setLevel(logging.INFO)
        self._chkpt = chkpt
        self._tensorboard = tensorboard
        if tensorboard:
            self._writer = SummaryWriter(log_dir=os.path.join(config.session_dir, ConfigFile.tensorboard_dirname))
        self._multigpu = multigpu

    def dump(self, path=None):
        """Dumps class instance as serialized pickle file
        Args:
            path (str): dumping path
        """
        if not path:
            path = os.path.join(self.config.session_dir, ConfigFile.trainer_filename)
        self._dump(path)

    def load(self, path):
        """Loads serialized file to initialize class instance

        Args:
            path (str): Path to file
        """
        return self._load(path)

    def _compute_loss(self, output, target):
        loss = self.criterion(output, target)
        return loss

    def _eval_metrics(self, data, target, output):
        return np.array([metric(output, target) for metric in self.metrics])

    @abstractmethod
    def _train_epoch(self, epoch, dataloader):
        """Run training on an epoch.

        Args:
            epoch (int): epoch number
            dataloader (core.dataloader._base.BaseDataLoader)
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch, dataloader):
        """Run validation on an epoch

        Args:
            epoch (int): epoch number
            dataloader (core.dataloader._base.BaseDataLoader)
        """
        raise NotImplementedError

    @abstractmethod
    def _image_callback(self, data, epoch, **kwargs):
        """Image dumping callback for tensorboard

        Args:
            data (torch.Tensor): images batch
            epoch (int): epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _embedding_callback(self, data, target, epoch, **kwargs):
        """Image dumping callback for tensorboard

        Args:
            data (torch.Tensor): images batch
            target (torch.Tensor): groundtruth
            epoch (int): epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _pr_curve_callback(self, data, target, epoch, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _histogram_callback(self, epoch):
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto', global_step=epoch)

    @validation
    def _callbacks(self, epoch, seed=None):
        """At the end of validation loop, flows through the different
        tensorboard callbacks that have been setup

        - Histogram callback: only depends on model and can always be ran
        - Image callback: must generate batch of images
        - Embedding and PR callbacks: must generate large subset for it to be
            meaningful

        Args:
            epoch (int): epoch number
            seed (int): random seed, allows to generate deterministically
                samples of data for inference and perform consistent comparison
                of behavior evolution through epochs
        """
        # TODO : implement callback selection in a cleaner way... la c'est du SALE
        # setup model for inference
        self.model.eval()

        with torch.no_grad():
            # Write histograms
            try:
                self._histogram_callback.__isabstractmethod__
            except AttributeError:
                self._histogram_callback(epoch)

            # Write images, skip if not implemented
            try:
                self._image_callback.__isabstractmethod__
            except AttributeError:
                # Generate small batch and run it through image callback
                data, target = zip(*self.dataloader.choices(k=8, seed=seed or self.seed))
                self._image_callback(data, target, epoch)

            # Generate larger batch, we use a flag to avoid generating it twice
            gen_flag = False

            # Write embeddings
            try:
                self._embedding_callback.__isabstractmethod__
            except AttributeError:
                data, target = zip(*self.dataloader.choices(k=300, seed=seed or self.seed))
                gen_flag = True
                self._embedding_callback(data, target, epoch)

            # Write precision/recall curves
            try:
                self._pr_curve_callback.__isabstractmethod__
            except AttributeError:
                if not gen_flag:
                    data, target = zip(*self.dataloader.choices(k=300, seed=seed or self.seed))
                self._pr_curve_callback(data, target, epoch)

    def _save_checkpoint(self, epoch, **kwargs):
        """Dumps checkpoint file in checkpoint directory

        Args:
            epoch (int): epoch number
        """
        filename = ConfigFile.checkpoints_format.format(epoch=epoch)
        path = os.path.join(self.config.session_dir,
                            ConfigFile.checkpoints_dirname, filename)
        self._logger.verbose("Saving checkpoint : {} ...".format(filename))

        arch = type(self.model).__name__
        state = {'arch': arch,
                 'epoch': epoch,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'model_class': self.model.__class__.__name__,
                 'optimizer_class': self.optimizer.__class__.__name__,
                 **kwargs}
        torch.save(state, path)

    def resume_checkpoint(self, epoch):
        """Loads checkpoint file and sets trainer up, ready for pursuing
        training from this point

        Args:
            epoch (int): epoch number
        """
        filename = ConfigFile.checkpoints_format.format(epoch=epoch)
        path = os.path.join(self.config.session_dir,
                            ConfigFile.checkpoints_dirname, filename)
        self._logger.verbose("Loading checkpoint : {}".format(filename))

        chkpt = torch.load(path)
        self.config.set_init_epoch(chkpt['epoch'] + 1)

        # load architecture params from checkpoint.
        if chkpt['model_class'] != self.config.model.__class__.__name__:
            warnings.warn("Architecture configuration given in config file is different from that of checkpoint")
        self.model.load_state_dict(chkpt['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        if chkpt['optimizer_class'] != self.optimizer.__class__.__name__:
            warnings.warn("Optimizer type given in config file is different from that of checkpoint. \n Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(chkpt['optimizer'])

        self._logger.verbose("Checkpoint loaded. Resume training from epoch {}".format(self.init_epoch))
        return chkpt

    def _setup_training(self, dataloader):
        """To be called at beginning of fit method to setup everything there
        is to setup before initiating training

        Args:
            dataloader (core.dataloader._base.BaseDataLoader)
        """
        # Setup logs frequency
        self._log_steps = int(np.sqrt(dataloader.batch_size))

        # Use parallel wrapper
        if self.multigpu and torch.cuda.device_count() > 1:
            self.set_model(torch.nn.DataParallel(self.model))

        # Move model to device
        self.set_model(self.model.to(self.device))

    def fit(self, dataloader=None, seed=None):
        """Main training method

        Args:
            dataloader (core.dataloader._base.BaseDataLoader): default is
                its own dataloader attribute but can provide external loader
            seed (int): random seed
        """
        # setup dataloader
        dataloader = dataloader or self.dataloader

        # Generate validation dataloader
        val_loader = dataloader.validation_loader()

        # Setup model and training params
        self._setup_training(dataloader)

        for epoch in range(self.init_epoch, self.epochs):
            # Init log dictionnary
            logs = {'epoch': epoch + 1}

            # Run training and validation loops
            train_logs = self._train_epoch(epoch=epoch, dataloader=dataloader)
            val_logs = self._valid_epoch(epoch=epoch, dataloader=val_loader)

            logs.update({**train_logs, **val_logs, 'lr': self.current_lr})

            # Print epoch review
            self._logger.verbose("".center(80, "*"))
            for key, value in logs.items():
                self._logger.verbose('\n    {:15s}: {}'.format(str(key).upper(), value))
            self._logger.verbose("".center(80, "*"))

            # Dump tensorboard logs
            if self.tensorboard:
                for key, value in logs.items():
                    self.writer.add_scalar(key, value, global_step=epoch)
                self._callbacks(epoch, seed)

            # Save checkpoint
            if self.chkpt:
                self._save_checkpoint(epoch)

            # Update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._config.model

    @property
    def criterion(self):
        return self._config.criterion

    @property
    def optimizer(self):
        return self._config.optimizer

    @property
    def metrics(self):
        return self._config.metrics

    @property
    def init_epoch(self):
        return self._config.init_epoch

    @property
    def epochs(self):
        return self._config.epochs

    @property
    def dataloader(self):
        return self._config.dataloader

    @property
    def config(self):
        return self._config

    @property
    def verbose(self):
        return self._verbose

    @property
    def chkpt(self):
        return self._chkpt

    @property
    def tensorboard(self):
        return self._tensorboard

    @property
    def writer(self):
        return self._writer

    @property
    def lr_scheduler(self):
        return self._config.lr_scheduler

    @property
    def seed(self):
        return self._config.seed

    @property
    def current_lr(self):
        if self.lr_scheduler:
            return self.lr_scheduler.get_lr()[0]
        else:
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    @property
    def multigpu(self):
        return self._multigpu

    def set_device(self, device):
        self._device = device

    def set_model(self, model):
        self._config.set_model(model)

    def set_verbose(self, verbose):
        self._verbose = verbose

    def set_tensorboard(self, tensorboard):
        self._tensorboard = tensorboard

    def set_chkpt(self, chkpt):
        self._chkpt = chkpt
