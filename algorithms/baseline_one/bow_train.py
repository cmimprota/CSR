# INPUT: -ver <version number>
#        -v <vocabulary filename>
#        -d <dataset filename>

# OUTPUT: model representation of the NN
import os
import pickle
import random
from argparse import ArgumentParser
from collections import Counter

import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.baseline_one.bow_model import BoWClassifier
import constants


############################### I N I T I A L I Z A T I O N     B E G I N ###############################


torch.manual_seed(1)

parser = ArgumentParser()
parser.add_argument("-d", choices=["train-short", "train-full"])
parser.add_argument("-v", type=str)
parser.add_argument("-ver", type=int)
args = parser.parse_args()

with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"{args.v}"), "rb") as inputfile:
    vocab = pickle.load(inputfile)

VOCAB_SIZE = len(vocab)
NUM_LABELS = 2
WORD_TO_INDEX = {}
for i in range(len(vocab)):
    WORD_TO_INDEX[vocab[i]] = i


def encode_tweets_as_vectors(file, label):
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        tweets = f.readlines()

    tweets_as_vectors = []
    for tweet in tweets:
        # TODO do the same pre-processing as in build-vocab-full (call it from separate script.py)
        words = tweet.split()
        occurances_of_words = list(Counter(words).items())
        tweet_vector = torch.zeros(len(vocab))
        # Do not iterate over every entry in vocab in order to set that position to 0 or occurances.
        # It will consume way to much processing time, that is why we have dictionary of words in vocab and their index
        # This way we can only iterate over sentences in tweet saving enormous processing time (factor of more than 10)
        for word, count in occurances_of_words:
            if word in WORD_TO_INDEX.keys():
                tweet_vector[WORD_TO_INDEX[word]] = count
        tweets_as_vectors.append((tweet_vector.view(1, -1), label))
    return tweets_as_vectors


############################### P R O G R A M     B E G I N ###############################


# We only use training datasets, no test ones (we split this script into test and train)!
# As training will take long, I don't want code breaking in testing (I want to save the middle result as a checkpoint)
if args.d == "train-short":
    parse_positive = "train_pos.txt"
    parse_negative = "train_neg.txt"
else:
    parse_positive = "train_pos_full.txt"
    parse_negative = "train_neg_full.txt"

all_tweets_as_vectors = []
all_tweets_as_vectors += encode_tweets_as_vectors(parse_positive, 1)
all_tweets_as_vectors += encode_tweets_as_vectors(parse_negative, 0)

# We do not want to train first on positive and then negative
# We should shuffle them for better results!
random.shuffle(all_tweets_as_vectors)


model = BoWClassifier(VOCAB_SIZE, NUM_LABELS)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
losses = []
for epoch in range(10):
    total_loss = 0
    for tweet_vector, label in all_tweets_as_vectors:
        model.zero_grad()
        log_probs = model(tweet_vector)
        # Do not forget to convert label into a vector!
        loss = loss_function(log_probs, torch.LongTensor([label]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)
joblib.dump(model, f"bow-model-{args.v}-vocab-version-{args.ver}.pkl", compress=9)
