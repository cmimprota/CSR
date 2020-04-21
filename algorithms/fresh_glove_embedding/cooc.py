# INPUT: -v <vocabulary filename>
#        -d <dataset filename> (defaults to short dataset)

# OUTPUT: cooc-<vocab>.pkl

# copied and adapted from https://github.com/dalab/lecture_cil_public/tree/master/exercises/ex6

from scipy.sparse import *    # this script needs scipy >= v0.15
import numpy as np
import pickle
import os
from argparse import ArgumentParser

import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ) # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import constants

parser = ArgumentParser()
parser.add_argument("-v", type=str, required=True, help="vocabulary (without path, without extension)")
parser.add_argument("-d", choices=["train-short", "train-full"], default="train-short")
args = parser.parse_args()


def main():
    # use a vocab without stemming or any modification
    with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"{args.v}.pkl"), "rb") as inputfile:
        vocab = pickle.load(inputfile)
    if isinstance(vocab, list):
        # convert to dict representation
        vocab = { vocab[i]: i for i in range(len(vocab)) }
    assert isinstance(vocab, dict)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    files_to_parse = ['train_pos_full.txt', 'train_neg_full.txt'] if args.d=="train-full" else ['train_pos.txt', 'train_neg.txt']
    for fn in files_to_parse:
        with open(os.path.join(constants.DATASETS_PATH, fn), "r") as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    
    curdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curdir, f'cooc__{args.d}__{args.v}.pkl'), 'wb') as f:
        pickle.dump(cooc, f)


if __name__ == '__main__':
    main()
