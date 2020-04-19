# INPUT: -v <vocabulary filename>
#        --full to run on full dataset 'train_pos/neg_full.txt', otherwise run on 'train_pos/neg.txt'

# OUTPUT: cooc.pkl

from scipy.sparse import *    # this script needs scipy >= v0.15
import numpy as np
import pickle
import os
from argparse import ArgumentParser
import constants

parser = ArgumentParser()
parser.add_argument("-v", type=str)
# parser.add_argument("--full", type=bool, action="store_false")
parser.add_argument("--full", type=bool, default=False)
args = parser.parse_args()


def main():
    # use a vocab without stemming or any modification
    with open(os.path.join(constants.VOCABULARIES_CUT_PATH, f"{args.v}"), "rb") as inputfile:
        vocab = pickle.load(inputfile)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    files_to_parse = ['train_pos_full.txt', 'train_neg_full.txt'] if args.full else ['train_pos.txt', 'train_neg.txt']
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
    with open('cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f)


if __name__ == '__main__':
    main()
