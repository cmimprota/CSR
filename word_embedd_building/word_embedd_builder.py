import os
import pickle
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

def create_WordEmbeddBuilder(method, **kwargs):
    """A kind of "factory" function to create WordEmbeddBuilder
    Args: 
        kwargs (dict): additional parameters passed through keyword arguments. Possible parameters:
            for method "fresh-glove":
                epochs
                ... (TODO)
    """
    from trivial import TrivialWEB
    
    if method == "trivial":
        return TrivialWEB()
    elif method == "fresh-glove":
        assert 'freq_threshold' in kwargs # TODO
        return FreshGloveWEB(**kwargs)
    else:
        raise ValueError("unrecognized cutting method")

class WordEmbeddBuilder: # (skl.base.TransformerMixin):
    """Abstract class for word-embedding building"""
    def __init__(self):
        super().__init__()
    def build_word_embedd(self, vocabulary, tweets, interm_dir=None):
        """
        Args: 
            vocabulary (iterable of string): the list of words for which we compute an embedding
            tweets (iterable of string): the (preprocessed) tweets to use to compute the word-embedding
            interm_dir (string): path to a directory where intermediary results can be written
                It can be assumed that `interm_dir` corresponds to the same preprocessing method as was used to get `tweets`
        Returns: 
            the embedding as np.ndarray of size (nb_words, embedding_dimension) 
                with the rows representing words in the same order as the input `vocabulary` list
        """
        raise NotImplementedError("WordEmbeddBuilder is a (fake) abstract class. Must use a subclass, e.g TrivialWEB")
