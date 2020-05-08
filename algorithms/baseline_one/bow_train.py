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


def encode_tweets_as_bow_vectors(file, label):
    """
    From dataset in the file loads all the tweets and encodes each of them as bow vector
    :param file: Filename in the twitter-datasets folder
    :param label: label of the the data in this file
    :return: List of Tuples (bow vector, label)
    """
    # We assume that each line is a single tweet
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        tweets = f.readlines()

    bow_vectors = []
    # BOW encodes whole sentences(tweets) as bow vector that will be used as input vector. Check the bottom for example
    # We need to use whole bow_vector as input layer and therefore it has same length that is equal to vocabulary size
    for tweet in tweets:
        # TODO do the same pre-processing as in build-vocab-full (call it from separate script.py)
        words = tweet.split()
        occurrences_of_words = list(Counter(words).items())
        bow_vector = torch.zeros(len(vocab), device=DEVICE)
        # Do not iterate over every entry in vocab in order to set that position to 0 or occurrences.
        # It will consume way to much processing time, that is why we have dictionary of words in vocab and their index
        # This way we can only iterate over sentences in tweet saving enormous processing time (factor of more than 10)
        for word, count in occurrences_of_words:
            if word in WORD_TO_INDEX.keys():
                bow_vector[WORD_TO_INDEX[word]] = count

        # Create tuple and append it to result while transforming bow_vector that is column vector into a row vector
        bow_vectors.append((bow_vector.view(1, -1), label))
    return bow_vectors


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
all_tweets_as_bow_vectors = []
all_tweets_as_bow_vectors += encode_tweets_as_bow_vectors(parse_positive, 1)
all_tweets_as_bow_vectors += encode_tweets_as_bow_vectors(parse_negative, 0)

# We do not want to train first on positive and then negative. We should shuffle them for better results!
random.shuffle(all_tweets_as_bow_vectors)

# Read the model description in bow_model.py
model = BoWClassifier(VOCAB_SIZE, NUM_LABELS)
model.to(DEVICE)

# BoWClassifier uses softmax on output - we have vector of log probabilities(one element for each class - in our case 2)

# Negative log likelihood function - input to our loss_function is log probabilities and the label
# By putting label, we define which position in output we want to optimize for and which class it represents
# In our case we will have log probabilities for negative class on position 0 and on position 1 for positive class
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Not needed but can be used for debugging
losses = []

# TODO: Consider to parametrize number of epochs as well
for epoch in range(20):
    # Good practice is to log the progress somehow (might be useful for running on a cluster)
    print(F"Running epoch {epoch}\n")

    # Accumulates the losses of the current epoch
    total_loss = 0
    for bow_vector, label in all_tweets_as_bow_vectors:
        # Always call when training the data
        model.zero_grad()
        log_probabilities = model(bow_vector)
        # Do not forget to convert label into a vector otherwise it will not work!!!
        loss = loss_function(log_probabilities, torch.LongTensor([label], device=DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

# Print losses of all epochs
print(losses)


############################### S A V I N G     B E G I N ###############################


save_model(model, f"{args.v}", f"{args.d}")


############################### B O W     E X A M P L E ###############################
"""
vocabulary:
"this"
"is"
"an"
"example"
"of"
"vocabulary"

linear layer (input size = length(vocabulary))

tweet_1 = "this this not"
tweet_2 = "example an is"

bow_vector_1 = [2 0 0 0 0 0] - we have 2 occurrences of word "this" which in vocabulary is at position 0. 
                               not is not in vocabulary so we just skip it. This is the input to the ANN 
bow_vector_2 = [0 1 1 1 0 0] - left for practice!

second approach: instead of putting number of occurrences just put 0 or 1 depending if it exists in the tweet or not
bow_vector_1 = [1 0 0 0 0 0] - in this case this appears and we just set at that position to 1
"""