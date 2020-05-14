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
    parser.add_argument("to_preproc", help="raw tweets dataset to preprocess (full path to txt file)")
    args = parser.parse_args()

    # extract base dataset name from the path, expected of the form `twitter-datasets/<base_data>.txt`
    base_data = os.path.splitext( os.path.basename(args.to_preproc) )[0]

    # create output dir if it doesn't exist yet
    outdir = os.path.join(constants.DATASETS_PATH, f"preprocessed__{args.meth}") # TODO: include eventual parameters in the dir name
    os.makedirs(outdir, exist_ok=True)
    
    with open(args.to_preproc) as f:
        tweets = f.readlines()
    
    DP = create_DataPreprocessor(method=args.meth)

    with open(os.path.join(outdir, f"{base_data}.txt"), 'w') as f:
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
        """
        Args:
            tweet (str): the raw tweet
        Returns:
            preprocessed_tweet (str): the preprocessed tweet as a string
        """
        raise NotImplementedError("DataPreprocessor is a (fake) abstract class. Must use a subclass, e.g NoopDP or XcbDP")

if __name__ == '__main__':
    main()
