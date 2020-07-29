"""Model definitions (one class per file) to define models."""
from .CharWordCNN import CharAndWordCNN

__all__ = ('CharWordCNN')


def get_model(name):
    if name == 'CharWordCNN':
        return CharAndWordCNN
    else:
        raise NotImplementedError("No model named \"%s\"!" % name)