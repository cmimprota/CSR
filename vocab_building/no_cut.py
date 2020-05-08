from vocab_builder import VocabBuilder, build_occ_dict
import os
import pickle

class NoCutVB(VocabBuilder):
    """
    Very naive and inefficient vocabulary-building method: don't cut anything, keep all words present in the data
    """
    def __init__(self):
        super().__init__()

    def build_vocab(self, tweets, interm_dir):
        # get the full occurrence dictionary
        # try loading it from interm_dir
        path_to_occ_dict = interm_dir and os.path.join(interm_dir, "full_occurrence_dict.pkl")
        if interm_dir is not None and os.path.exists(path_to_occ_dict):
            with open(path_to_occ_dict, "rb") as f:
                occ_dict = pickle.load(f)
        else:
            occ_dict = build_occ_dict(tweets)
            # save intermediary result
            with open(path_to_occ_dict, "wb") as f:
                pickle.dump(occ_dict, f)

        print(type(occ_dict))
        return list(occ_dict.keys())
