from word_embedd_builder import WordEmbeddBuilder

class FreshGloveWEB(WordEmbeddBuilder):
    """
    Builds a word-embedding using Glove
    """
    def __init__(self):
        super().__init__()
    def build_word_embedd(self, vocabulary, tweets, interm_dir=None):
        raise NotImplementedError
