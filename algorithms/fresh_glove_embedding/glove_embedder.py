# INPUT: -v <vocabulary filename> (use file `cooc-<vocab>.pkl` in the same dir)
#        -d <dataset filename> (defaults to short dataset)'

# OUTPUT: embeddings-<vocab>.npz

# copied and adapted from https://github.com/dalab/lecture_cil_public/tree/master/exercises/ex6

from scipy.sparse import *
import numpy as np
import pickle
import random
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-v", "--vocab", type=str, required=True, help="vocabulary (without path, without extension)")
parser.add_argument("-d", "--dataset", choices=["train-short", "train-full"], default="train-short")
parser.add_argument("--nmax", type=int, default=100)
parser.add_argument("--embedding_dim", type=int, default=20)
parser.add_argument("--eta", type=float, default=0.001)
parser.add_argument("--alpha", type=float, default=0.75)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

def main():

    print("loading cooccurrence matrix")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'cooc__{args.d}__{args.v}.pkl'), 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings");
    print("cooc shape 0: ", cooc.shape[0], "cooc shape 1: ", cooc.shape[1])
    embedding_dim = args.embedding_dim
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = args.eta
    alpha = args.alpha

    epochs = 20

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x

    curdir = os.path.dirname(os.path.abspath(__file__))
    np.savez(os.path.join(curdir, f'embeddings__{args.d}__{args.v}.npz'), xs=xs, ys=ys)


if __name__ == '__main__':
    main()
    