# INPUT: vocab-to-pickle.pkl (previously pickled vocabulary)
#        -o path/to/output/file.txt (output path. default: vocab-to-pickle.txt)
# OUTPUT: vocab-to-pickle.txt or user-specified by -o

# copied and adapted from kaggle helper files

import pickle
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("input", type=str, help="the .pkl file to unpickle")
parser.add_argument("-o", metavar="output", type=str, help="path to the output .txt file")
args = parser.parse_args()

file_name, file_ext = os.path.splitext(args.input)
assert file_ext == ".pkl", "File to unpickle must be a pickle file (with extension .pkl)"
if args.o is None:
    args.o = file_name + '.txt'

def main():
    with open(args.input, "rb") as f:
        vocab = pickle.load(f)

    with open(args.o, 'w') as f:
        # TODO: drop support for dictionary representation inside pkl files
        if isinstance(vocab, dict):
            # sort dictionary by value and iterate
            for k, v in sorted(vocab.items(), key=lambda item: item[1]):
                f.write(str(k) + "\n")
        else:
            assert isinstance(vocab, list)
            f.write('\n'.join(vocab))



if __name__ == '__main__':
    main()
