# INPUT: -d <dataset filename>
#        -meth <preprocessing method>

# OUTPUT: txt file of preprocessed tweets

# Note: contrary to word-embeddings, data-preprocessing methods are deterministic, i.e
# the output for a given tweet is independent of the other tweets in the dataset

import os
import pickle
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ) # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import constants

def main():
    parser = ArgumentParser()
    parser.add_argument("--meth", choices=["xcb", "noop"], default="xcb")
    parser.add_argument("to_preproc", help="raw tweets dataset to preprocess") #, choices=["test_data", "train_neg_full", "train_neg", "train_pos_full", "train_pos"])
    args = parser.parse_args()

    # create output dir if it doesn't exist yet
    outdir = os.path.join(constants.DATASETS_PATH, f"preprocessed__{args.meth}") # TODO: include eventual parameters in the dir name
    os.makedirs(outdir, exist_ok=True)
    
    with open(os.path.join(constants.DATASETS_PATH, f"{args.to_preproc}.txt")) as f:
        tweets = f.readlines()
    
    DP = create_DataPreprocessor(method=args.meth)

    with open(os.path.join(outdir, f"{args.to_preproc}.txt"), 'w') as f:
        for tweet in tqdm(tweets):
            res = DP.preprocess(tweet)
            f.write(res + "\n")

def create_DataPreprocessor(method):
    """A kind of "factory" function to create DataPreprocessor
    Args: 
        method ("xcb" or "noop"): method to use for preprocessing tweets
    """
    from noop import NoopDP
    from xcb import XcbDP

    if method == "xcb":
        return XcbDP()
    elif method == "noop":
        return NoopDP()
    else:
        raise ValueError("unrecognized preprocessing method")

class DataPreprocessor: # (skl.base.TransformerMixin):
    """Absract class for data preprocessing"""
    def __init__(self):
        super().__init__()
    def preprocess(self, tweet):
        raise NotImplementedError("DataPreprocessor is a (fake) abstract class. Must use a subclass, e.g NoopDP or XcbDP")

if __name__ == '__main__':
    main()
