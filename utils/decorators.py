import os
import sys
import time
import copy
from functools import wraps
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)
import utils.IOHandler as io


def setseed(fn):
    @wraps(fn)
    def wrapper(*args, seed=None, **kwargs):
        if seed:
            np.random.seed(seed)
        return fn(*args, seed=seed, **kwargs)
    return wrapper


def timeit(fn):
    def timed(*args, **kwargs):
        ts = time.time()
        result = fn(*args, **kwargs)
        te = time.time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', fn.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (fn.__name__, (te - ts) * 1000))
        return result
    return timed


def selfaccepts(*types):
    def check_accepts(fn):
        assert len(types) == fn.__code__.co_argcount - 1

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                    "arg %r does not match %s" % (a, t)
            return fn(self, *args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    return check_accepts


def accepts(*types):
    def check_accepts(fn):
        assert len(types) == fn.__code__.co_argcount

        @wraps(fn)
        def wrapper(*args, **kwargs):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                    "arg %r does not match %s" % (a, t)
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    return check_accepts


def serializable(cls):

    def dump(self, path):
        """Dumps class instance as serialized pickle file
        Args:
            path (str): dumping path
        """
        buffer = copy.deepcopy(self)
        io.save_dill(path, buffer)

    @classmethod
    def load(cls, path):
        """Loads serialized file to initialize class instance

        Args:
            path (str): Path to file
        """
        buffer = io.load_dill(path)
        if not isinstance(buffer, cls):
            raise TypeError("Loaded serialized file is not of proper class")
        return copy.deepcopy(buffer)

    cls._dump = dump
    cls._load = load
    return cls
