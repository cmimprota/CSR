from data_preprocessor import DataPreprocessor

class NoopDP(DataPreprocessor):
    def __init__(self):
        super().__init__()
    def preprocess(self, tweet):
        return tweet
