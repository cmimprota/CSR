# INPUT: vocab-to-pickle.txt (choose any cut vocabulary; in fact it works for arbitrary text files)
#        -o path/to/output/file.pkl (output path. default: vocab-to-pickle.pkl)
#        --format "list" or "dict" to choose the format (default: list)
# OUTPUT: vocab-to-pickle.pkl or user-specified by -o

# copied and adapted from kaggle helper files

import pickle
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("input", type=str, help="the .txt file to pickle")
parser.add_argument("-o", metavar="output", type=str, help="path to the output .pkl file")
parser.add_argument("--format", type=str, help="pickle the vocab as a list or as a dict", choices=["list", "dict"], default="list")
args = parser.parse_args()

file_name, file_ext = os.path.splitext(args.input)
assert file_ext == ".txt", "File to pickle must be a text file (with extension .txt)"
if args.o is None:
    args.o = file_name + '.pkl'

def main():
    with open(args.input) as f:
        # TODO: drop support for dictionary representation inside pkl files
        if args.format == "list": # list representation l[idx]=word
            vocab = f.read().splitlines()
        else: # dict representation {word: idx}
            vocab = dict()
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx

    with open(args.o, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
