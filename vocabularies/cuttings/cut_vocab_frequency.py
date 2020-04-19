# INPUT: -v full/vocab_full_[test/train/both].pkl (choose any full vocabulary)
#        -f <frequency number> (minimum frequency for a word so that it is kept)

# OUTPUT: cut/cut-vocab-frequency<frequency number>.pkl
# TYPE: list(tuple)<(word, frequency)>

import pickle
from argparse import ArgumentParser
import constants
import os

parser = ArgumentParser()
parser.add_argument("-v", choices=["test", "train-short", "train-full", "test-and-train-short", "test-and-train-full"])
parser.add_argument("-f", type=int)
args = parser.parse_args()

with open(os.path.join(constants.VOCABULARIES_FULL_PATH, f"{args.v}.pkl"), "rb") as inputfile:
    full_vocab = pickle.load(inputfile)

# Select all words occurring at least f times.
cut_vocab = [word for (word, count) in full_vocab if count > args.f]

with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"cut-vocab-{args.v}-frequency-{args.f}.pkl"), "wb") as outputfile:
    pickle.dump(cut_vocab, outputfile)
