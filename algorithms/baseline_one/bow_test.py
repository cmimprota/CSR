# INPUT: -m <model foldername>

# OUTPUT: submission file
import os
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import torch

import constants
from algorithms.helpers import load_model, save_submission, load_vocabulary

parser = ArgumentParser()
parser.add_argument("-v", type=str)
parser.add_argument("-m", type=str)
args = parser.parse_args()


vocab = load_vocabulary(f"{args.m}")

word_to_index = {}
for i in range(len(vocab)):
    word_to_index[vocab[i]] = i


# No need to parametrize as we have a single test_data file
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

model = load_model(f"{args.m}")
label_predictions = []
with torch.no_grad():
    for tweet_vector in tweets_as_vectors:
        log_probs = model(tweet_vector)
        probabilities = np.exp(log_probs)
        label_predictions.append(-1 if probabilities[0][0] > 0.5 else 1)


############################### S A V I N G     B E G I N ###############################


save_submission(label_predictions, f"{args.m}")
