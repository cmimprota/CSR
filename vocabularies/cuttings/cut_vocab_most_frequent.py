# INPUT: -v full/vocab_full_[test/train/both].pkl (choose any full vocabulary)
#        -n <frequency number> (top n words based on frequency that we will keep)

# OUTPUT: cut/cut-vocab-frequency<frequency number>.pkl
# TYPE: list(tuple)<(word, frequency)>

import pickle
from argparse import ArgumentParser
import constants
import os

parser = ArgumentParser()
parser.add_argument("-v", choices=["test", "train-short", "train-full", "test-and-train-short", "test-and-train-full",
                                   "nodup-test-and-train-full"])
parser.add_argument("-n", type=int)
args = parser.parse_args()

with open(os.path.join(constants.VOCABULARIES_FULL_PATH, f"{args.v}.pkl"), "rb") as inputfile:
    full_vocab = pickle.load(inputfile)

# Select the n most frequent words.
full_vocab.sort(key=lambda p: p[1], reverse=True)
cut_vocab = [word for (word, count) in full_vocab[:args.n]]

with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"cut-vocab-{args.v}-{args.n}-most-frequent.pkl"), "wb") as outputfile:
    pickle.dump(cut_vocab, outputfile)

with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"cut-vocab-{args.v}-{args.n}-most-frequent.txt"), "w") as outputfile:
    outputfile.write('\n'.join(cut_vocab))
