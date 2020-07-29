# INPUT: -m <model - foldername from results folder for model that you want to load>

# OUTPUT: submission file
import os
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import torch

import constants
from algorithms.helpers import load_model, save_submission, load_vocabulary

parser = ArgumentParser()
parser.add_argument("-m", type=str, help="model - foldername from results folder for model that you want to load")
parser.add_argument("-d", type=str, help="put nodup-test-and-train-full if using preprocessed")
args = parser.parse_args()

# Reads the configuration_log.csv file to find the exact vocabulary that was used for the selected model
vocab = load_vocabulary(f"{args.m}")

WORD_TO_INDEX = {}
for i in range(len(vocab)):
    WORD_TO_INDEX[vocab[i]] = i


# No need to add this as argument of script as we have a single test_data file
if args.d == "nodup-test-and-train-full":
    with open(os.path.join(constants.DATASETS_PATH, "test_data.txt"), "r") as f:
        tweets = f.readlines()
else:
    # Use the same pre-processing as in build-vocab-full (use created dataset and not function)
    with open(os.path.join(constants.DATASETS_PATH, "test_data.txt"), "r") as f:
        tweets = f.readlines()

# Read encode_tweets_as_bow_vectors in bow_train.py to understand how tweets are encoded
# Exactly the same approach is used here without labels as test do not have it
all_tweets_as_bow_vectors = []
for tweet in tweets:
    words = tweet.split()
    occurrences_of_words = list(Counter(words).items())
    bow_vector = torch.zeros(len(vocab))
    for word, count in occurrences_of_words:
        if word in WORD_TO_INDEX.keys():
            bow_vector[WORD_TO_INDEX[word]] = count
    all_tweets_as_bow_vectors.append(bow_vector.view(1, -1))

# Loads the 'trained_model' from the trained_model.pkl file from the selected folder
model = load_model(f"{args.m}")
label_predictions = []
with torch.no_grad():
    for bow_vector in all_tweets_as_bow_vectors:
        log_probs = model(bow_vector)
        # Softmax function is applied on the 2 output nodes and it gives us log probabilities so we need to exp them
        probabilities = np.exp(log_probs)
        # In the bow_train.py we are using position 0 for negative class by passing 0 label to the NNNLoss function
        label_predictions.append(-1 if probabilities[0][0] > 0.5 else 1)


############################### S A V I N G     B E G I N ###############################


# Saves the predicted classes from 'label_predictions' into the submission.csv file in the selected folder
save_submission(label_predictions, f"{args.m}")
