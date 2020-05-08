from vocab_builder import VocabBuilder, build_occ_dict
import os
import pickle

class TopKFrequencyVB(VocabBuilder):
    """
    Cut to only keep the `k` most frequent words in the data
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def build_vocab(self, tweets, interm_dir=None):
        # get the full occurrence dictionary
        # try loading it from interm_dir
        path_to_occ_dict = interm_dir and os.path.join(interm_dir, "full_occurrence_dict.pkl")
        if interm_dir is not None and os.path.exists(path_to_occ_dict):
            with open(path_to_occ_dict, "rb") as f:
                occ_dict = pickle.load(inputfile)
        else:
            occ_dict = build_occ_dict(tweets)
            # save intermediary result
            with open(path_to_occ_dict, "wb") as f:
                pickle.dump(occ_dict, f)

        # sort dictionary by value
        sortedl = [ k for k, v in sorted(occ_dict.items(), key=lambda item: item[1]) ]
        
        # Select the self.k most frequent words.
        return sortedl[:self.k]
