from data_preprocessor import DataPreprocessor

class NoopDP(DataPreprocessor):
    def __init__(self):
        super().__init__()
    def preprocess_tweet(self, tweet):
        return tweet
