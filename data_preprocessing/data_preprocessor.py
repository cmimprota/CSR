def create_DataPreprocessor(method):
    """A kind of "factory" function to create DataPreprocessor
    Args: 
        method (str): method to use for preprocessing tweets
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
    def preprocess_tweet(self, tweet):
        """
        Args:
            tweet (str): the raw tweet
        Returns:
            preprocessed_tweet (str): the preprocessed tweet as a string
        """
        raise NotImplementedError("DataPreprocessor is a (fake) abstract class. Must use a subclass, e.g NoopDP or XcbDP")
