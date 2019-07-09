# Engine : Training Configs and Trainer

## Training Configs (`./config_file.py`)

Contains a serializable class `ConfigFile` gathering all needed specifications  about the experiment one wishes to run (losses, optimizer, epochs, dataloader, ...). Those are specifications wrt the modelisation.

## Trainer

### `./trainer.py`

Contains a serializable class `Trainer` gathering high level specification wrt the training (device, verbose, logs, ...). Those are technical specification, agnostic to the modelisation.

`Trainer` class is endowed with a `fit` method allowing to run the whole experiment.

### `./neural_conditioner.py`

Wraps trainer class by explicitely implementing :

- Training and validation loops
- Metrics evaluation (optional)
- Loss computation (optional)
- Callbacks (optional)

---
## TODO
- [ ] Add `yacs` config format option
