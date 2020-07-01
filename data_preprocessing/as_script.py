# OUTPUT: txt file of preprocessed tweets

# Note: contrary to e.g word-embeddings, data-preprocessing methods are deterministic, i.e
# the output for a given tweet is independent of the other tweets in the dataset

import os
import pickle
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

RESULTS_SUBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from data_preprocessor import create_DataPreprocessor, DataPreprocessor

def main():
    parser = ArgumentParser()
    parser.add_argument("--method", "-m", choices=["xcb", "noop"], default="xcb")
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("to_preproc", help="raw tweets dataset to preprocess (full path to txt file)")
    args = parser.parse_args()
    
    if args.output is not None:
        outdir = os.path.dirname(args.output)
        outfile = args.output
    else:
        # extract base dataset name from the path, expected of the form `twitter-datasets/<base_data>.txt`
        base_data = os.path.splitext( os.path.basename(args.to_preproc) )[0]
        outdir = os.path.join(RESULTS_SUBDIR, f"preprocessed__{args.method}")
        outfile = os.path.join(outdir, f"{base_data}.txt")
    
    # create output dir if it doesn't exist yet
    os.makedirs(outdir, exist_ok=True)

    with open(args.to_preproc) as f:
        tweets = ( tweet.rstrip() for tweet in f )
    
    DP = create_DataPreprocessor(method=args.method)

    with open(outfile, 'w') as f:
        for tweet in tqdm(tweets):
            res = DP.preprocess_tweet(tweet)
            f.write(res + "\n")

if __name__ == '__main__':
    main()
