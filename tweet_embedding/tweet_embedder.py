import os
import pickle
import numpy as np

from naive_methods import MeanVectorTE, PaddedConcatTE

def create_TweetEmbedder(vocab, word_embedd, method, max_tweet_length=None, pad_mode=None):
    """A kind of "factory" function to create TweetEmbedder"""
    if method == "mean":
        return MeanVectorTE(vocab, word_embedd)
    elif method == "raw":
        return PaddedConcatTE(vocab, word_embedd, max_tweet_length, pad_mode)
    else:
        raise ValueError("unrecognized tweet-embedding method")


class TweetEmbedder: # (skl.base.TransformerMixin):
    """Embed tweets into a vector using a word-embedding
    Attributes:
        word_to_index (dict {str: int}): inverse mapping of vocab, maps word to index
        word_embedd (np.ndarray): word-embedding with shape (VOCAB_SIZE, EMBEDDING_DIM)
        VOCAB_SIZE (int)
        EMBEDDING_DIM (int)
        OUTPUT_DIM (int) (actually a getter/property. Present for convenience. Cannot be used inside the implementation of `embed_tweet`!)
        oov_vector (np.ndarray): fixed vector representing out-of-vocabulary words, with shape (EMBEDDING_DIM,)
    Parameters:
        vocab (list of str): vocab[i] is the word i
        word_embedd: (see Attributes)
    """
    def __init__(self, vocab, word_embedd):
        super().__init__()
        self.vocab = vocab
        self.word_embedd = word_embedd
        
        self.word_to_index = {}
        for i in range(len(vocab)):
            self.word_to_index[vocab[i]] = i

        self.VOCAB_SIZE = len(vocab)
        self.EMBEDDING_DIM = word_embedd.shape[1]

        self.oov_vector = np.zeros((self.EMBEDDING_DIM,)) # TODO: maybe use a different policy for unknown words
        
    @property
    def OUTPUT_DIM(self):
        if not self._output_dim: # if cached, return cache; otherwise compute
            dummy_embedd_tw = self.embed_tweet(["hello", "world"]) # to get the shape of the output
            assert len(dummy_embedd_tw.shape) == 1
            self.output_dim_ = dummy_embedd_tw.shape[0]
        return self._output_dim

    def embed_tweet(self, tweet):
        """Represent the tweet as a vector
        Args:
            tweet (list of string): the tweet as a list of words
        Returns: 
            embedded_tweet (np.ndarray): with shape (OUTPUT_DIM,)
        """
        raise NotImplementedError("TweetEmbedder is a (fake) abstract class. Must use a subclass, e.g MeanVectorTE")
