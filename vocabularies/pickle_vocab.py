# INPUT: vocab-to-pickle.txt (choose any cut vocabulary; in fact it works for arbitrary text files)
#        -o path/to/output/file.pkl (output path. default: vocab-to-pickle.pkl)
# OUTPUT: vocab-to-pickle.pkl or user-specified by -o

# copied and adapted from kaggle helper files

import pickle
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("input", type=str, help="the .txt file to pickle", required=True)
parser.add_argument("-o", type=str, help="path to the output .pkl file")
args = parser.parse_args()

file_name, file_ext = os.path.splitext(args.input)
assert file_ext == ".txt", "File to pickle must be a text file (with extension .txt)"
if args.o is None:
    args.o = file_name + '.pkl'

def main():
    vocab = dict()
    with open(args.input) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(args.o, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
