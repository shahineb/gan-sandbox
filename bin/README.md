# Experiments

An experiment is initialized as a directory containing all the needed information to reproduce the so told engine.

```
session_name
├── config.pickle
├── chkpt
│   └── chkpt_xxx.pth
├── runs
│   └── tensorboard logs
├── scores
└── logs.txt
```

It contains :

- `config.pickle`: a serialized training configuration file containing all information about how training should be performed (losses, optimizer, epochs, dataloader, ...)
- `chkpt`: directory where model weights checkpoints are stored as pytorch format files
- `runs`: directory where tensorboard logs are stored
- `scores`: directory where model evaluation dataframes are stored as csv files
- `logs.txt`: output training logs

(see more in `tutorial.ipynb`)

## Run an experiment

__Setup :__

Access `dashboard.ipynb` and go through the notebook to pick a session name, define model architecture, dataloader, training specs. Save all under session directory as specified above.

__Training :__

Run `python train.py --session  --gpu_id --resume --multigpu > session_name/logs.txt` where :
  - `--session`: name of the training session directory
  - `--gpu_id`: optional, allows to switch gpu (default: `0`)
  - `--resume`: optional, allows to resume training from `chkpt_xx.pth` (default: `None`)
  - `--multigpu`: optional, allows to parallelize computation if more than 1 device are available (default: `False`)

*Warning : make sure the proper classes are imported in `train.py`, otherwise you won't be able to load the model*

__Monitoring :__

Go to session directory and run `tensorboard --port 6008 --logdir runs/`
