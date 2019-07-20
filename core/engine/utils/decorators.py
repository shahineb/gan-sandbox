import torch
from functools import wraps


def training(fn):
    """Ensures model is in training mode
    Args:
        fn (method)
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.model.training:
            self.model.train()
        return fn(self, *args, **kwargs)
    return wrapper


def validation(fn):
    """Ensures model is in validation mode
    Args:
        fn (method)
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.model.training:
            self.model.eval()
        with torch.no_grad():
            output = fn(self, *args, **kwargs)
        return output
    return wrapper
