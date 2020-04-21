# INPUT: -v <vocabulary filename>
#        -d <dataset filename> (defaults to short dataset)
#        -meth <sentence embedding method> (defaults to mean vector of word embeddings)

# OUTPUT: npy of embedded tweets: embedded-datasets__<training_dataset>__<vocab>/embedded_<dataset_filename>.npy

import os
import pickle
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ) # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import constants

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # to fetch word-embeddings and write tweet-embeddings to the same dir as the embedding method (GLOVE)

def main():
    parser = ArgumentParser()
    parser.add_argument("-v", type=str, required=True, help="vocabulary (without path, without extension) (to choose which embedding to use)")
    parser.add_argument("-d", choices=["train-short", "train-full"], default="train-short", help="training dataset (to choose which embedding to use)")
    parser.add_argument("-meth", choices=["mean", "raw"], default="mean")
    parser.add_argument("to_embed", help="tweets to embed", choices=["test_data", "train_neg_full", "train_neg", "train_pos_full", "train_pos"])
    args = parser.parse_args()

    # create output dir if it doesn't exist yet
    outdir = os.path.join(CURRENT_DIR, f"embedded-datasets__{args.d}__{args.v}__{args.meth}")
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(constants.DATASETS_PATH, f"{args.to_embed}.txt")) as f:
        tweets = f.readlines()

    ET = create_EmbedTweets(vocab_fn=args.v, dataset_fn=args.d, method=args.meth)
    
    # dummy_embedd_tw = ET.embed(["hello", "world"]) # to get the shape of the output
    # res = np.empty((len(tweets), *dummy_embedd_tw.shape))
    # for tweetnb, tweet in enumerate(tweets):
    #     res[tweetnb] = ET.embed(tweet)
    res = np.array([ET.embed(tweet) for tweet in tweets])

    np.save(os.path.join(outdir, f"embedded_{args.to_embed}.npy"), res)


def create_EmbedTweets(vocab_fn, dataset_fn, method):
    """A kind of "factory" function to create EmbedTweets from vocab and (training) dataset filenames
    Args: 
        vocab_fn (string): vocabulary filename (without path, without extension)
        dataset_fn ("train-short" or "train-full"): training dataset
        method ("mean" or "raw"): method to use for embedding tweets
    """
    # read vocabulary
    with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"{vocab_fn}.pkl"), "rb") as inputfile:
        vocab = pickle.load(inputfile)
    # read embedding
    with np.load(os.path.join(CURRENT_DIR,f"embeddings__{dataset_fn}__{vocab_fn}.npz")) as data:
        xs = data['xs']
        ys = data['ys']
    return EmbedTweets(vocab, xs, ys, method)


class EmbedTweets:
    """Embed tweets into a vector using a word-embedding
    Attributes:
        word_to_index (dict {str: int}): inverse mapping of vocab
        xys (np.ndarray): concatenation of xs and ys with shape (VOCAB_SIZE, EMBEDDING_DIM)
            in some models we ignore the x/y separation and take the embedding of word i to be xys[i]=(xs[i],ys[i])
        VOCAB_SIZE (int)
        X_EMBEDDING_DIM (int)
        EMBEDDING_DIM (int)
        oov_vector (np.ndarray): fixed vector representing out-of-vocabulary words, with shape (EMBEDDING_DIM,)
    Parameters:
        vocab (list of str): vocab[i] is the word i
        xs (np.ndarray): x embedding with shape (VOCAB_SIZE, X_EMBEDDING_DIM)
        ys (np.ndarray): y embedding
        method ("mean" or "raw"): method to use for embedding tweets
    """
    def __init__(self, vocab, xs, ys, method):
        self.vocab = vocab
        self.xs = xs
        self.ys = ys
        assert method in ["mean", "raw"]
        self.method = method
        
        self.word_to_index = {}
        for i in range(len(vocab)):
            self.word_to_index[vocab[i]] = i

        self.xys = np.concatenate((xs, ys), axis=1)
        self.VOCAB_SIZE = len(vocab)
        self.X_EMBEDDING_DIM = xs.shape[1]
        self.EMBEDDING_DIM = 2*xs.shape[1]

        self.oov_vector = np.zeros((self.EMBEDDING_DIM,)) # TODO: maybe use a different policy for unknown words

    def embed(self, tweet):
        if self.method == "mean":
            return self.embed_tweet_by_mean_vector(tweet)
        elif self.method == "raw":
            return self.embed_tweet_by_padded_concat(tweet)
        else: raise ValueError("method should be 'mean' or 'raw'")

    def embed_tweet_by_mean_vector(self, tweet):
        """Represent the tweet by the mean of its word embeddings
        Returns: 
            embedded_tweet (np.ndarray): with shape (EMBEDDING_DIM,)
        """
        l = [ self.xys[self.word_to_index[word]] if word in self.vocab else self.oov_vector for word in tweet ]
        return np.mean(np.array(l), axis=0)

    def embed_tweet_by_padded_concat(self, tweet, max_tweet_length=30, pad_mode=0.):
        """Represent the tweet by the concatenation of its word embeddings ("raw" embedding)
        - If tweet is longer than max_tweet_length, truncate it
        - If tweet is shorter than max_tweet_length, pad with pad_mode (by default, 0's)
        Rk: this method can be really heavy and is mainly there for convenience and testing
        Args:
            pad_mode (str or function or float): the padding mode for https://numpy.org/doc/stable/reference/generated/numpy.pad.html
                if float, pad with constant value (over all dimensions of the embedding)
        Returns:
            embedded_tweet (np.ndarray): with shape (EMBEDDING_DIM*max_tweet_length,)
        """
        l = [ self.xys[self.word_to_index[word]] if word in self.vocab else self.oov_vector for word in tweet ]
        concated = np.concatenate(np.array(l), axis=1)
        if len(tweet) < max_tweet_length:
            diff = self.EMBEDDING_DIM*( max_tweet_length-len(tweet) )
            if isinstance(pad_mode, str):
                concated = np.pad(concated, (0,diff), mode=pad_mode)
            elif isinstance(pad_mode, (int, float)):
                concated = np.pad(concated, (0,diff), constant_values=pad_mode)
            else: raise ValueError
        assert concated.shape == (max_tweet_length*self.EMBEDDING_DIM,)
        return concated


if __name__ == '__main__':
    main()
