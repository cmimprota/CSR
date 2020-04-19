# INPUT: -ver <version number>      # TODO CHECK IF I NEED THIS
#        -d <dataset filename>      # TODO CHECK IF I NEED THIS
#        -v <vocabulary filename>
#        -m <model filename>

# OUTPUT: submission file
import os
import pickle
from collections import Counter
from argparse import ArgumentParser

import joblib
import numpy as np
import torch

from algorithms.baseline_one.bow_model import BoWClassifier
import constants

parser = ArgumentParser()
#parser.add_argument("-d", choices=["train-short", "train-full"])
parser.add_argument("-v", type=str)
#parser.add_argument("-ver", type=int)
parser.add_argument("-m", type=str)
args = parser.parse_args()


with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"{args.v}"), "rb") as inputfile:
    vocab = pickle.load(inputfile)

# Maybe can be handled differently, filling the torch.zeros is the challenge
word_to_index = {}
for i in range(len(vocab)):
    word_to_index[vocab[i]] = i


with open(os.path.join(constants.DATASETS_PATH, "test_data.txt"), "r") as f:
    tweets = f.readlines()

tweets_as_vectors = []
for tweet in tweets:
    # TODO do the same pre-processing as in build-vocab-full (call it from separate script.py)
    words = tweet.split()
    occurances_of_words = list(Counter(words).items())
    tweet_vector = torch.zeros(len(vocab))
    for word, count in occurances_of_words:
        if word in word_to_index.keys():
            tweet_vector[word_to_index[word]] = count
    tweets_as_vectors.append(tweet_vector.view(1, -1))

model = joblib.load(f"{args.m}")

with open("submission.csv", "w") as f:
    f.write("Id,Prediction\n")
    i = 0;
    with torch.no_grad():
        for tweet_vector in tweets_as_vectors:
            log_probs = model(tweet_vector)
            probabilities = np.exp(log_probs)
            i += 1
            f.write(f"{i},{-1 if probabilities[0][0] > 0.5 else 1}\n")
