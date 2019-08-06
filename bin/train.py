import os
import sys
from absl import app, flags
import torch

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(base_dir)


"""
Make sure all classes required by config file are imported
>>>
"""
from core.dataloader import CelebALoader
from core.models import VAE, Discriminator, Generator, modules
from core.engine import ConfigFile, NCTrainer, GANTrainer, MixGANTrainer
################################################################################


FLAGS = flags.FLAGS


def main(argv):
    del argv
    session_dir = os.path.join(ConfigFile.bin_dir, FLAGS.session)

    # Load config file
    config_path = os.path.join(session_dir, ConfigFile.pickle_filename)
    config = ConfigFile.load(config_path)

    # Setup trainer instance
    params = {'config': config,
              'device': torch.device(FLAGS.device),
              'verbose': FLAGS.verbose,
              'chkpt': FLAGS.chkpt,
              'tensorboard': FLAGS.tb,
              'multigpu': FLAGS.multigpu}
    trainer = MixGANTrainer(**params)

    # Load checkpoint
    if FLAGS.resume:
        trainer.resume_checkpoint(int(FLAGS.resume))
    trainer._logger.info(trainer.config)

    # Train model
    trainer.fit()


if __name__ == '__main__':
    flags.DEFINE_string('session', None, "name of the training session to run")
    flags.DEFINE_string('resume', None, "name of the checkpoint file to load")
    flags.DEFINE_string('device', "cuda", "training device in {'cpu', 'cuda'}")
    flags.DEFINE_integer('verbose', 1, "verbosity level in {0, 1}")
    flags.DEFINE_bool('chkpt', True, "if True, saves checkpoint file at the end epochs")
    flags.DEFINE_bool('tb', True, "if True, dumps tensorboard logs")
    flags.DEFINE_bool('multigpu', False, "if True, parallelizes computation across available devices")
    app.run(main)
