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
args = parser.parse_args()

# Reads the configuration_log.csv file to find the exact vocabulary that was used for the selected model
vocab = load_vocabulary(f"{args.m}")

WORD_TO_INDEX = {}
for i in range(len(vocab)):
    WORD_TO_INDEX[vocab[i]] = i


# Same device needs to be used when instantiating tensors as for model. Exception thrown otherwise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is the maximum number of words that we will take into consideration from a single tweet
MAX_NO_OF_WORDS = 20


# No need to add this as argument of script as we have a single test_data file
with open(os.path.join(constants.DATASETS_PATH, "test_data.txt"), "r") as f:
    tweets = f.readlines()

all_tweets_as_bow_gru_matrices = []
for tweet in tweets:
    # TODO do the same pre-processing as in build-vocab-full (call it from separate script.py)
    words = tweet.split()
    current_number_of_words = 0
    bow_gru_matrix = []
    for word in words:
        # The dimension of matrix must always be MAX_NO_OF_WORDS x length(vocabulary)
        if not current_number_of_words < MAX_NO_OF_WORDS:
            break

        # Words not in vocabulary are skipped
        if word in WORD_TO_INDEX.keys():
            # Row of the input matrix for GRU has a single 1
            bow_gru_row_vector = [0.0] * len(vocab)
            #bow_gru_row_vector = torch.zeros(len(vocab), device=DEVICE)
            bow_gru_row_vector[WORD_TO_INDEX[word]] = 1.0
            bow_gru_matrix.append(bow_gru_row_vector)
            current_number_of_words += 1

    # Pad the matrix with 0 row vectors in case that matrix is not full already
    while current_number_of_words < MAX_NO_OF_WORDS:
        bow_gru_row_vector = [0.0] * len(vocab)
        bow_gru_matrix.append(bow_gru_row_vector)
        current_number_of_words += 1
        #bow_gru_matrix.append(torch.zeros(len(vocab), device=DEVICE))

    # Create tuple and append it to result
    all_tweets_as_bow_gru_matrices.append(torch.tensor([bow_gru_matrix], device=DEVICE))

# Loads the 'trained_model' from the trained_model.pkl file from the selected folder
model = load_model(f"{args.m}")
label_predictions = []
with torch.no_grad():
    for bow_gru_matrix in all_tweets_as_bow_gru_matrices:
        predictions = torch.sigmoid(model(bow_gru_matrix))
        label_predictions.append(1 if predictions > 0.5 else -1)

############################### S A V I N G     B E G I N ###############################


# Saves the predicted classes from 'label_predictions' into the submission.csv file in the selected folder
save_submission(label_predictions, f"{args.m}")
