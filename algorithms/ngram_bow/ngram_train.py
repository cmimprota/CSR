from nltk import ngrams
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import device
import os
import pickle
import constants
from algorithms.ngram_bow.ngram_bow_model import NGramLanguageModeler
import random

torch.manual_seed(1)

parser = ArgumentParser()
parser.add_argument("-d", choices=["train-short", "train-full"], help="dataset - choose train-short or train-full")
parser.add_argument("-v", type=str, help="vocabulary - filename in the folder cut")
parser.add_argument("-n", type=int, help="ngrams size")
args = parser.parse_args()

with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"{args.v}"), "rb") as inputfile:
    vocab = pickle.load(inputfile)

# TODO Check this out
EMBEDDING_DIM = 10

CONTEXT_SIZE = args.n
VOCAB_SIZE = len(vocab)
NUM_LABELS = 2
WORD_TO_INDEX = {}
for i in range(len(vocab)):
    WORD_TO_INDEX[vocab[i]] = i

# Same device needs to be used when instantiating tensors as for model. Exception thrown otherwise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context_vector(context):
    idxs = [WORD_TO_INDEX[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long, device=DEVICE)


# TODO check what to do with labels
def encode_tweets_as_ngram_bow_vectors(file, label):
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        tweets = f.readlines()
    ngram_bow_vectors = []
    for tweet in tweets:
        # TODO do the same pre-processing as in build-vocab-full (call it from separate script.py)
        words = tweet.split()
        # ngrams_bow = ngrams(words.split(), CONTEXT_SIZE)
        for i in range(CONTEXT_SIZE, len(words) - CONTEXT_SIZE):
            context = []
            for j in range(CONTEXT_SIZE, 0, -1):
                context.append(words[i - j])
            for k in range(1, CONTEXT_SIZE + 1, 1):
                context.append(words[i + j])
            target = words[i]
            ngram_bow_vectors.append((make_context_vector(context), torch.tensor([WORD_TO_INDEX[target]], dtype=torch.long, device=DEVICE)))

    return ngram_bow_vectors


if args.d == "train-short":
    parse_positive = "train_pos.txt"
    parse_negative = "train_neg.txt"
else:
    parse_positive = "train_pos_full.txt"
    parse_negative = "train_neg_full.txt"

# Labels are chosen as 0 and 1 and correspond to the position of its probability in output vector (look loss_function)
all_tweets_as_ngram_bow_vectors = []
all_tweets_as_ngram_bow_vectors += encode_tweets_as_ngram_bow_vectors(parse_positive, 1)
all_tweets_as_ngram_bow_vectors += encode_tweets_as_ngram_bow_vectors(parse_negative, 0)

# We do not want to train first on positive and then negative. We should shuffle them for better results!
random.shuffle(all_tweets_as_ngram_bow_vectors)

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_function = nn.NLLLoss()
losses = []

for epoch in range(20):
    # Good practice is to log the progress somehow (might be useful for running on a cluster)
    print(F"Running epoch {epoch}\n")

    # Accumulates the losses of the current epoch
    total_loss = 0
    for context, target in all_tweets_as_ngram_bow_vectors:
        # Always call when training the data
        model.zero_grad()
        log_probabilities = model(context)
        # Do not forget to convert label into a vector otherwise it will not work!!!
        loss = loss_function(log_probabilities, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

# Print losses of all epochs
print(losses)
