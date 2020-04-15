# INPUT: -d twitter-datasets/[test/train/both] (can choose between just test, train, both)

# OUTPUT: full/vocab_full_[test/train/both].pkl
# TYPE: list(tuple)<(word, frequency)>

import pickle
from argparse import ArgumentParser
from collections import Counter
import constants
import os

parser = ArgumentParser()
parser.add_argument("-d", choices=["test", "train-short", "train-full", "test-and-train-short", "test-and-train-full"])
args = parser.parse_args()

files_to_parse = []

if args.d == "test":
    files_to_parse.append("test_data.txt")
elif args.d == "train-short":
    files_to_parse.append("train_pos.txt")
    files_to_parse.append("train_neg.txt")
elif args.d == "train-full":
    files_to_parse.append("train_pos_full.txt")
    files_to_parse.append("train_neg_full.txt")
elif args.d == "test-and-train-short":
    files_to_parse.append("train_pos.txt")
    files_to_parse.append("train_neg.txt")
    files_to_parse.append("test_data.txt")
elif args.d == "test-and-train-full":
    files_to_parse.append("train_pos_full.txt")
    files_to_parse.append("train_neg_full.txt")
    files_to_parse.append("test_data.txt")

words = []
for file in files_to_parse:
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        words += [word for word in f.read().split()]

vocab = list(Counter(words).items())
with open(os.path.join(constants.VOCABULARIES_FULL_PATH, f"{args.d}.pkl"), "wb") as f:
    pickle.dump(vocab, f)
