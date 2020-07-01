def build_occ_dict(tweets):
    """An auxiliary function that builds the occurrence dictionary
    Args:
        tweets (list of string): the data
    Returns:
        occ_dict (dict): dictionary of word occurrences {word: count}
    """
    from collections import Counter
    # words = [*tweet.split() for tweet in tweets] # nope (https://stackoverflow.com/a/41251957)
    words = [] # list of words with redundancy
    for tweet in tweets:
        words += tweet.split()
    occ_dict = Counter(words)
    return occ_dict
    
def create_VocabBuilder(method, **kwargs):
    """A kind of "factory" function to create VocabBuilder
    Args: 
        method (string): method to use for preprocessing tweets
        kwargs (dict): additional parameters passed through keyword arguments. Possible parameters:
            'freq_threshold' (float) for method "freq-thresh"
            'k' (int) for method "top-k-freq"
    """
    from no_cut import NoCutVB
    from frequency_threshold import FrequencyThresholdVB
    from top_k_frequency import TopKFrequencyVB
    # from tfidf import TfidfVB

    if method == "no-cut":
        return NoCutVB()
    elif method == "freq-thresh":
        assert 'freq_threshold' in kwargs
        return FrequencyThresholdVB(freq_thresh=kwargs['freq_threshold'])
    elif method == "top-k-freq":
        assert 'k' in kwargs
        return TopKFrequencyVB(k=kwargs['k'])
    # elif method == "tfidf":
    #     return TfIdfVB()
    else:
        raise ValueError("unrecognized cutting method")

class VocabBuilder: # (skl.base.TransformerMixin):
    """Abstract class for (cut) vocabulary building"""
    def __init__(self):
        super().__init__()
    def build_vocab(self, tweets, interm_dir=None):
        """
        Args:
            tweets (list of string): the (preprocessed) tweets to use to build the vocabulary
            interm_dir (string): path to a directory where intermediary results can be written
                It can be assumed that `interm_dir` corresponds to the same preprocessing method as was used to get `tweets`
        Returns:
            vocab (list of string): vocabulary as list of words
        """
        raise NotImplementedError("VocabBuilder is a (fake) abstract class. Must use a subclass, e.g FrequencyThresholdVB or TopKFrequencyVB")
