# OUTPUT: - vocabulary as text file
#         - also save to file an intermediate result (to avoid recomputing in case of subsequent failure):
#               occurrence count of the full vocabulary as a pkl of dict {word: count}

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
    parser.add_argument("--meth", choices=["no-cut", "freq-thresh", "top-k-freq"], required=True, help="cutting method")
    parser.add_argument("--freq", type=float, help="for frequency-threshold, the frequency threshold")
    parser.add_argument("-k", type=int, help="for top-k-frequency, the number of words to keep")
    parser.add_argument("dataset", help="preprocessed tweets dataset to use (full path to txt file)")
    # parser.add_argument("--all", action="store_true",  # default to False
    #     help="use all datasets in the specified data directory")
    # TODO: allow using all data from a given `preprocessed__<preproc_meth>/` dir`
    args = parser.parse_args()

    import warnings
    if args.meth == "freq-thresh":
        assert args.freq is not None, "parameter args.freq is required when cutting based on frequency-threshold"
        if args.k is not None: warnings.warn("parameter args.k was provided but will be ignored")
    elif args.meth == "top-k-freq":
        assert args.k is not None, "parameter args.k is required when cutting based on top-k-frequency"
        if args.freq is not None: warnings.warn("parameter args.freq was provided but will be ignored")

    # extract things from the path, expected of the form `twitter-datasets/preprocessed__<preproc_meth>/<base_data>.txt`
    preproc_meth = os.path.basename(os.path.dirname(args.dataset)).split("preprocessed__")[1]
    base_data = os.path.splitext( os.path.basename(args.dataset) )[0]
    # infer a pretty name for the preprocessed dataset
    dataset_name = f"{preproc_meth}__{base_data}"
    print(f"inferred pretty name dataset_name: {dataset_name}")

    # create output dir if it doesn't exist yet
    outdir = os.path.join(constants.VOCABULARIES_PATH, dataset_name)
    os.makedirs(outdir, exist_ok=True) 

    with open(args.dataset) as f:
        tweets = f.readlines() # (these are preprocessed tweets)

    VB = create_VocabBuilder(method=args.meth, freq_thresh=args.freq, k=args.k)

    vocab = VB.build_vocab(tweets, interm_dir=outdir)

    if args.meth == "no-cut":
        outfilename = args.meth
    elif args.meth == "freq-thresh":
        outfilename = f"{args.meth}__{args.freq}"
    elif args.meth == "top-k-freq":
        outfilename = f"{args.meth}__{args.k}"
    else:
        raise ValueError("unrecognized cutting method")

    with open(os.path.join(outdir, f"{outfilename}.txt"), 'w') as f:
        for word in tqdm(vocab):
            f.write(word + "\n")
    with open(os.path.join(outdir, f"{outfilename}.pkl"), 'wb') as f:
        pickle.dump(vocab, f)

def build_occ_dict(tweets):
    """
    Args:
        tweets (list of string): the data
    Returns:
        occ_dict (dict): dictionary of word occurrences {word: count}
    """
    from collections import Counter
    # words = [*tweet.split() for tweet in tweets] # list of words with redundancy
    words = []
    for tweet in tweets:
        words += tweet.split()
    occ_dict = Counter(words)
    return occ_dict
    
def create_VocabBuilder(method, **kwargs):
    """A kind of "factory" function to create VocabBuilder
    Args: 
        method ("xcb" or "noop"): method to use for preprocessing tweets
        kwargs (dict): additional parameters passed through keyword arguments. Possible parameters:
            'freq_thresh' (float) for method "freq_thresh"
            'k' (int) for method "top_k_freq"

    """
    from no_cut import NoCutVB
    from frequency_threshold import FrequencyThresholdVB
    from top_k_frequency import TopKFrequencyVB
    # from tfidf import TfidfVB

    if method == "no-cut":
        return NoCutVB()
    elif method == "freq_thresh":
        assert 'freq_thresh' in kwargs
        return FrequencyThresholdVB(freq_thresh=kwargs['freq_thresh'])
    elif method == "top_k_freq":
        assert 'k' in kwargs
        return TopKFrequencyVB(k=kwargs['k'])
    else:
        raise ValueError("unrecognized cutting method")

class VocabBuilder: # (skl.base.TransformerMixin):
    """Absract class for (cut) vocabulary building"""
    def __init__(self):
        super().__init__()
    def build_vocab(self, tweets, interm_dir):
        """
        Args:
            tweets (list of string): the (preprocessed) tweets to use to build the vocabulary
            interm_dir (string): path to a directory where intermediary results can be written
                It can be assumed that `interm_dir` corresponds to the same preprocessing method as was used to get `tweets`
        Returns:
            vocab (list of string): vocabulary as list of words
        """
        raise NotImplementedError("VocabBuilder is a (fake) abstract class. Must use a subclass, e.g FrequencyThresholdVB or TopKFrequencyVB")

if __name__ == '__main__':
    main()
