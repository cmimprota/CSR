from vocab_builder import VocabBuilder, build_occ_dict
import os
import pickle

class FrequencyThresholdVB(VocabBuilder):
    """
    Cut to only keep words that have more than `freq_thresh` occurrences in the data
    """
    def __init__(self, freq_thresh):
        super().__init__()
        self.freq_thresh = freq_thresh

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

        # Select all words occurring at least self.freq_thresh times.
        return [word for word in occ_dict if occ_dict[word] >= self.freq_thresh]
