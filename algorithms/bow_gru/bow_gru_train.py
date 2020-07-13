# INPUT: -v <vocabulary - filename in the folder cut>
#        -d <dataset - choose train-short or train-full>

# OUTPUT: model representation of the NN
import os
import pickle
import random
from argparse import ArgumentParser
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import device

import constants
from algorithms.baseline_one.bow_model import BoWClassifier
from algorithms.bow_gru.bow_gru_model import BoWGRUClassifier
from algorithms.helpers import save_model


############################### I N I T I A L I Z A T I O N     B E G I N ###############################


torch.manual_seed(1)

parser = ArgumentParser()
parser.add_argument("-d", choices=["train-short", "train-full"], help="dataset - choose train-short or train-full")
parser.add_argument("-v", type=str, help="vocabulary - filename in the folder cut")
args = parser.parse_args()


# It is expected that you have previously created a vocabulary using some .py script from cuttings folder
# This means that you can selected the created .pkl from the cut folder by passing its filename as args.v
with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"{args.v}"), "rb") as inputfile:
    vocab = pickle.load(inputfile)

# Vocabulary is in my case ordered dictionary word->occurrences_of_word
# Use it to create a dictionary word->index. Key is word and value is position of word in original vocabulary
# This is needed in order to have faster access when encoding tweets (otherwise encoding for short dataset needs 1h)
VOCAB_SIZE = len(vocab)
NUM_LABELS = 2
WORD_TO_INDEX = {}
for i in range(len(vocab)):
    WORD_TO_INDEX[vocab[i]] = i

# Same device needs to be used when instantiating tensors as for model. Exception thrown otherwise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is the maximum number of words that we will take into consideration from a single tweet
MAX_NO_OF_WORDS = 20


def encode_tweets_as_bow_gru_matrices(file, label):
    """
    From dataset in the file loads all the tweets and encodes each of them as bow_gru_matrix
    :param file: Filename in the twitter-datasets folder
    :param label: label of the the data in this file
    :return: List of Tuples (bow_gru_matrix, label)
    """
    # We assume that each line is a single tweet
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        tweets = f.readlines()

    bow_gru_matrices = []
    # BOW encodes whole sentences(tweets) as bow vector that will be used as input vector. Check the bottom for example
    # We need to use whole bow_vector as input layer and therefore it has same length that is equal to vocabulary size
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
        bow_gru_matrices.append((torch.tensor([bow_gru_matrix], device=DEVICE), label))
    return bow_gru_matrices


############################### T R A I N I N G     B E G I N ###############################


# We only use training datasets, no test ones (we split this script into test and train)!
# As training will take long, I don't want code breaking in testing (I want to save the middle result as a checkpoint)
if args.d == "train-short":
    parse_positive = "train_pos.txt"
    parse_negative = "train_neg.txt"
else:
    parse_positive = "train_pos_full.txt"
    parse_negative = "train_neg_full.txt"

# Labels are chosen as 0 and 1 and correspond to the position of its probability in output vector (look loss_function)
all_tweets_as_bow_gru_matrices = []
all_tweets_as_bow_gru_matrices += encode_tweets_as_bow_gru_matrices(parse_positive, 1.0)
all_tweets_as_bow_gru_matrices += encode_tweets_as_bow_gru_matrices(parse_negative, 0.0)

# We do not want to train first on positive and then negative. We should shuffle them for better results!
random.shuffle(all_tweets_as_bow_gru_matrices)

# Read the model description in bow_gru_model.py
model = BoWGRUClassifier(input_size=VOCAB_SIZE,
                         hidden_dim=256,
                         output_dim=1,
                         n_layers=2,
                         bidirectional=True,
                         dropout=0.25)

model = model.to(DEVICE)

# BoWClassifier uses BCELoss on output - we have single node in output layer
# When using labels we should put 0 for negative class and 1 for positive class in the 1d tensor
# Because the label is either 0 or 1. Sigmoid followed by a BCELoss. But the output of model is a 1d scalar, no sigmoid.
# In that case negative class is for output < 0.5 and positive for > 0.5
# (if we put positive class in sigmoid as 0 then it is opposite)
optimizer = optim.Adam(model.parameters(), lr=0.1)
loss_function = nn.BCEWithLogitsLoss()
loss_function = loss_function.to(DEVICE) 

# Not needed but can be used for debugging
losses = []

# TODO: Consider to parametrize number of epochs as well
for epoch in range(1):
    # TODO: Training for one epoch took from 12:30 - 15:00
    # Good practice is to log the progress somehow (might be useful for running on a cluster)
    print(F"Running epoch {epoch}\n")

    # Accumulates the losses of the current epoch
    total_loss = 0
    for bow_gru_matrix, label in all_tweets_as_bow_gru_matrices:
        # Always call when training the data
        model.zero_grad()
        prediction = model(bow_gru_matrix).squeeze(1)
        # Do not forget to convert label into a vector otherwise it will not work!!!
        loss = loss_function(prediction, torch.tensor([label], device=DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

# Print losses of all epochs
print(losses)


############################### S A V I N G     B E G I N ###############################


save_model(model, f"{args.v}", f"{args.d}")


############################### B O W    G R U   E X A M P L E ###############################
"""
-In case of using GRU, we need to define our input as a matrix MxN (input size = length(vocabulary))
-N is equal to the length of vocabulary as usual BOW approach
-M is the number of words that we will take into consideration. 
 For example, if define M=20 whis will mean that out of all words that tweet consist of we consider only first 20.
 If tweet has more than 20 words, the remaining will be ignored.
 If tweet has less than 20 words, the missing ones will be filled with 0 vectors.
 Every vector is encoded as one-hot vector meaning it has 1 on single position and all others are 0

Example:

vocabulary:
"this"
"is"
"an"
"example"
"of"
"vocabulary"

MAX_NO_OF_WORDS=20

GRU (input size = MAX_NO_OF_WORDS x length(vocabulary))

tweet_1 = "this this not"
tweet_2 = "example an is"

bow_gru_matrix_1  = [[1 0 0 0 0 0] [1 0 0 0 0 0] [0 0 0 0 0 0] [0 0 0 0 0 0] .... [0 0 0 0 0 0]] 
Each word is encoded as one-hot vector. Since "this" is in vocabulary is at position 0, we have vector [1 0 0 0 0 0]
"this" appears on first two positions in tweet_1 and in the matrix notation this corresponds to first two rows
"not" is not in vocabulary so we skip. Since we predefine matrix with 20 rows, rest are 0. This is input to GRU or LSTM

bow_gru_matrix_2  = [[0 0 0 1 0 0] [0 0 1 0 0 0] [0 1 0 0 0 0] [0 0 0 0 0 0] .... [0 0 0 0 0 0]] - left for practice!
"""

