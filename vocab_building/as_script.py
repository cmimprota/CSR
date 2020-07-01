# OUTPUT: - (cut) vocabulary as a list of words
#         - also saves .pkl of occurrence dictionary of word occurrences {word: count}

import os
import pickle
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

RESULTS_SUBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from vocab_builder import create_VocabBuilder, VocabBuilder

def main():
    parser = ArgumentParser()
    parser.add_argument("--method", "-m", choices=["no-cut", "freq-thresh", "top-k-freq"], required=True, help="cutting method")
    parser.add_argument("--freq", "-f", type=float, help="for frequency-threshold, the frequency threshold")
    parser.add_argument("-k", type=int, help="for top-k-frequency, the number of words to keep")
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("dataset", help="preprocessed tweets dataset to use (full path to txt file)")
    # parser.add_argument("--all", action="store_true",  # default to False
    #     help="use all datasets in the specified data directory")
    # TODO: allow using all data from a given `preprocessed__<preproc_meth>/` dir`
    args = parser.parse_args()

    import warnings
    if args.method == "freq-thresh":
        assert args.freq is not None, "parameter args.freq is required when cutting based on frequency-threshold"
        if args.k is not None: warnings.warn("parameter args.k was provided but will be ignored")
    elif args.method == "top-k-freq":
        assert args.k is not None, "parameter args.k is required when cutting based on top-k-frequency"
        if args.freq is not None: warnings.warn("parameter args.freq was provided but will be ignored")

    if args.output is not None:
        outdir = os.path.dirname(args.output)
        outfile = args.output
    else:
        # extract things from the path, expected of the form `twitter-datasets/preprocessed__<preproc_meth>/<base_data>.txt`
        preproc_meth = os.path.basename(os.path.dirname(args.dataset)).split("preprocessed__")[1]
        base_data = os.path.splitext( os.path.basename(args.dataset) )[0]
        # infer a pretty name for the preprocessed dataset
        dataset_name = f"{preproc_meth}__{base_data}"
        print(f"inferred pretty name dataset_name: {dataset_name}")
        outdir = os.path.join(RESULTS_SUBDIR, dataset_name)

        if args.method == "no-cut":
            outfilename = args.method
        elif args.method == "freq-thresh":
            outfilename = f"{args.method}__{args.freq}"
        elif args.method == "top-k-freq":
            outfilename = f"{args.method}__{args.k}"
        else:
            raise ValueError("unrecognized cutting method")
        outfile = os.path.join(outdir, outfilename)

    # create output dir if it doesn't exist yet
    os.makedirs(outdir, exist_ok=True)

    with open(args.dataset) as f:
        tweets = ( tweet.rstrip() for tweet in f ) # (these are preprocessed tweets)

    VB = create_VocabBuilder(method=args.method, freq_threshold=args.freq, k=args.k)

    vocab = VB.build_vocab(tweets, interm_dir=outdir)
    
    with open(f"{outfile}.txt", 'w') as f:
        for word in tqdm(vocab):
            f.write(word + "\n")
    with open(f"{outfile}.pkl", 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == '__main__':
    main()
