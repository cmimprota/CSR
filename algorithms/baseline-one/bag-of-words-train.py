# INPUT: -ver <version number>
#        -v <vocabulary filename>
#        -d <dataset filename>

# OUTPUT: model representation of the NN
import os
import pickle
import random
from argparse import ArgumentParser
from collections import Counter
import numpy as np

import constants


parser = ArgumentParser()
parser.add_argument("-d", choices=["train-short", "train-full"])
parser.add_argument("-v", type=str)
parser.add_argument("-ver", type=int)
args = parser.parse_args()


with open(os.path.join(constants.VOCABULARIES_FULL_PATH, f"{args.v}.pkl"), "rb") as inputfile:
    vocab = pickle.load(inputfile)


def encode_tweets_as_vectors(file, label):
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        tweets = f.readlines()

    tweets_as_vectors = []
    for tweet in tweets:
        # TODO do the same pre-processing as in build-vocab-full (call it from separate script.py)
        words = tweet.split()
        occurances_of_words = list(Counter(words).items())
        tweet_vector = []
        for word in vocab:
            # similar as if word in (item[0] for item in occurances_of_words):
            match = [item for item in occurances_of_words if item[0] == word]
            if match is not None:
                # number of occurances instead of 1
                tweet_vector.append(match[1])
            else:
                tweet_vector.append(0)
        tweets_as_vectors.append((tweet_vector, label))
    return np.asarray(tweets_as_vectors)


# We only use training datasets, no test ones.
# We split as training will take long and I don't want something breaking while test
if args.d == "train-short":
    parse_positive = "train_pos.txt"
    parse_negative = "train_neg.txt"
else:
    parse_positive = "train_pos_full.txt"
    parse_negative = "train_neg_full.txt"

all_tweets_as_vectors = []
all_tweets_as_vectors.append(encode_tweets_as_vectors(parse_positive, 1))
all_tweets_as_vectors.append(encode_tweets_as_vectors(parse_negative, 0))

random.shuffle(all_tweets_as_vectors)