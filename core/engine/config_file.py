import os
import sys
import time
import torch

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
import config
import utils.IOHandler as io
from utils.decorators import serializable


@serializable
class ConfigFile:
    """Utility class gathering all needed information about a training session
    in order to ensure its reproducibility

    Attributes:
        _session_name (str): quite explicit
        _session_dir (str): full path to session directory
        _model (core.models._base.BaseNet): model used in this training session
        _criterion (callable): objective function used for training
        _optimizer (torch.optim.Optimizer): training optimizer
        _metrics (list[callable]): metrics used to assess model performances along training
        _dataloader (core.dataloader._base.BaseDataLoader): loader for training
        _init_epoch (int): initial epoch
        _epochs (int): number of training epochs
        _lr_scheduler (callable): callable to update learning rate through epochs
        _seed (int): random seed

    Static Attributes:
        bin_dir (str): naming of directory where training sessions are to take place
        pickle_filename (str): naming of serialized config file
        checkpoints_dirname (str): naming of subdirectory with model checkpoints
        checkpoints_format (str): naming of checkpoints files
        tensorboard_dirname (str): naming of subdirectory with tensorboard logs
        scores_dirname (str): naming of subdirectory with session scores
        trainer_filename (str): naming of serialized trainer file
        SEED (int): value of default random seed
    """
    bin_dir = config.bin_dir
    pickle_filename = "config.pickle"
    checkpoints_dirname = "chkpt"
    checkpoints_format = "chkpt_{epoch:03d}.pth"
    tensorboard_dirname = "runs"
    scores_dirname = "scores"
    trainer_filename = "trainer.pickle"
    SEED = 73

    def __init__(self, session_name=None, model=None, criterion=None,
                 optimizer=None, metrics=None, dataloader=None, epochs=None,
                 lr_scheduler=None, seed=None, **kwargs):
        self._session_name = session_name
        if model:
            assert issubclass(model.__class__, torch.nn.Module), "Invalid argument specified for model"
        self._model = model
        self._session_dir = os.path.join(ConfigFile.bin_dir, session_name)
        # TODO : review assertion verification
        self._criterion = criterion
        if optimizer:
            assert issubclass(optimizer.__class__, torch.optim.Optimizer), "Optimizer specified is not valid"
        self._optimizer = optimizer
        self._metrics = metrics
        self._dataloader = dataloader
        self._init_epoch = 0
        self._epochs = epochs
        if lr_scheduler:
            assert callable(lr_scheduler)
        self._lr_scheduler = lr_scheduler
        self._seed = seed or ConfigFile.SEED
        self._kwargs = kwargs

    def __repr__(self):
        output = "\n".join([" : ".join([key, str(value)]) for (key, value) in self.__dict__.items() if key not in ["_model"]])
        output = "\n".join([super(ConfigFile, self).__repr__(), output])
        return output

    def setup_session(self, overwrite=False, timestamp=False):
        """Sets up training session directory creating all default directories
        and files needed
        Args:
            overwrite (bool): if True, overwrites existing directory (default: False)
            timestamp (bool): if True, adds timestamp to directory name (default: False)
        """
        session_name = self.session_name
        if timestamp:
            session_name = session_name + "_" + time.strftime("%Y%m%d-%H%M%S")
        io.mkdir(session_name, ConfigFile.bin_dir, overwrite)
        session_dir = os.path.join(ConfigFile.bin_dir, session_name)
        io.mkdir(ConfigFile.checkpoints_dirname, session_dir)
        io.mkdir(ConfigFile.tensorboard_dirname, session_dir)
        io.mkdir(ConfigFile.scores_dirname, session_dir)
        ConfigFile._write_gitignore(session_dir)

    def dump(self, path=None):
        """Dumps class instance as serialized pickle file
        Args:
            path (str): dumping path
        """
        if not path:
            path = os.path.join(self.session_dir, ConfigFile.pickle_filename)
        self._dump(path)

    @classmethod
    def load(cls, path):
        """Loads serialized file to initialize class instance

        Args:
            path (str): Path to file
        """
        return cls._load(path)

    @staticmethod
    def _write_gitignore(dir_path):
        """Generates .gitignore file in specified directory to ignore all but
        gitignore file

        Args:
            dir_path (str): path to directory
        """
        with open(os.path.join(dir_path, ".gitignore"), "w") as f:
            f.write("*\n!.gitignore")

    @property
    def session_name(self):
        return self._session_name

    @property
    def session_dir(self):
        return self._session_dir

    @property
    def model(self):
        return self._model

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def metrics(self):
        return self._metrics

    @property
    def dataloader(self):
        return self._dataloader

    @property
    def init_epoch(self):
        return self._init_epoch

    @property
    def epochs(self):
        return self._epochs

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def seed(self):
        return self._seed

    @property
    def kwargs(self):
        return self._kwargs

    def set_session_name(self, session_name):
        self._session_name = session_name
        self._session_dir = os.path.join(ConfigFile.bin_dir, session_name)

    def set_model(self, model):
        assert issubclass(model.__class__, torch.nn.Module), "Invalid argument specified for model"
        self._model = model

    def set_criterion(self, criterion):
        self._criterion = criterion

    def set_optimizer(self, optimizer):
        assert issubclass(optimizer.__class__, torch.optim.Optimizer), "Optimizer specified is not valid"
        self._optimizer = optimizer

    def set_metrics(self, metrics):
        self._metrics = metrics

    def set_dataloader(self, dataloader):
        self._dataloader = dataloader

    def set_init_epoch(self, init_epoch):
        self._init_epoch = init_epoch

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_lr_scheduler(self, lr_scheduler):
        assert issubclass(lr_scheduler.__class__, torch.optim.lr_scheduler._LRScheduler)
        self._lr_scheduler = lr_scheduler

    def set_seed(self, seed):
        self._seed = seed

    def set_kwargs(self, **kwargs):
        self._kwargs = kwargs

    def update_kwargs(self, **kwargs):
        self._kwargs.update(kwargs)

    def add_metric(self, metric):
        self._metrics.append(metric)
