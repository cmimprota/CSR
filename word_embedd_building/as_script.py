import os
import pickle
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

RESULTS_SUBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from word_embedd_builder import *

def main():
    parser = ArgumentParser()
    parser.add_argument("--method", "-m", choices=["fresh-glove", "trivial"], required=True, help="word-embedding method")
    parser.add_argument("--epochs", type=int, default=3) # very low default
    parser.add_argument("--embedd_dim", type=int, default=20) # very low default
    parser.add_argument("vocabulary", help="the set of words to embed (full path to txt file)")
    parser.add_argument("dataset", help="preprocessed tweets dataset to use (full path to txt file)")
    # parser.add_argument("--all", action="store_true",  # default to False
    #     help="use all datasets in the specified data directory")
    # TODO: allow using all data from a given `preprocessed__<preproc_meth>/` dir`
    args = parser.parse_args()

    # import warnings
    # if args.method == "fresh-glove":
    #     assert args.epochs is not None, "parameter args.epochs is required when using fresh-glove method"
    #     assert args.embedd_dim is not None, "parameter args.embedd_dim is required when using fresh-glove method"
    
    if args.output is not None:
        outdir = os.path.dirname(args.output)
        outfile = args.output
    else:

        import time
        vocab_and_dataset = f"word_embeddings__{time.time()}" # just the timestamp... 
        # TODO: but the vocabulary is an important information to keep, because the rows are ordered w.r.t the order of words in that vocabulary file!!
        print(f"inferred pretty name vocab_and_dataset: {vocab_and_dataset}")
        outdir = os.path.join(RESULTS_SUBDIR, vocab_and_dataset)

        if args.method == "trivial":
            outfilename = args.method
        elif args.method == "fresh-glove":
            outfilename = f"{args.method}__epochs{args.epochs}__dim{args.embedd_dim}"
        else:
            raise ValueError("unrecognized word-embedding method")
        outfile = os.path.join(outdir, outfilename)

    # create output dir if it doesn't exist yet
    os.makedirs(outdir, exist_ok=True)

    with open(args.vocabulary) as vocabfile:
        vocab = (word.rstrip() for word in vocabfile)
    with open(args.dataset) as tweetsfile:
        tweets = (tweet.rstrip() for tweet in tweetsfile) # (these are preprocessed tweets)

    WEB = create_WordEmbeddBuilder(method=args.method, epochs=args.epochs, embedd_dim=args.embedd_dim) #, **args)
    word_embedding = WEB.build_word_embedd(vocab, tweets, interm_dir=None) # TODO: interm_dir

    # np.savez(f"{outfile}.npz"), xs=xs, ys=ys)
    np.save(f"{outfile}.npy"), word_embedding)
    
if __name__ == '__main__':
    main()
