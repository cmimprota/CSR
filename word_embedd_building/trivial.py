from word_embedd_builder import WordEmbeddBuilder

class TrivialWEB(WordEmbeddBuilder):
    """
    Associates each word to a single scalar number according to its frequency (corresponds to Naive Bayes)
    """
    def __init__(self):
        super().__init__()

    def build_word_embedd(self, vocabulary, tweets, interm_dir=None):
        """
        Args: 
            tweets (iterable of string): the (preprocessed) tweets to use to compute the word-embedding
            vocabulary (iterable of string): the list of words for which we compute an embedding
            interm_dir (string): path to a directory where intermediary results can be written
                It can be assumed that `interm_dir` corresponds to the same preprocessing method as was used to get `tweets`
        Returns: 
            the embedding as np.ndarray of size (nb_words, embedding_dimension) 
                with the rows representing words in the same order as the input `vocabulary` list
            Here, embedding_dimension = 1
        """
        EMBEDD_DIM = 1
        VOCAB_SIZE = len(vocabulary)
        # extremely dumb implementation but whatever
        rev_vocab = {word: idx for (idx, word) in enumerate(vocabulary)}
        out = np.zeros((VOCAB_SIZE,))
        for tweet in tweets:
            for word in tweet:
                out[rev_vocab[word]] += 1
        return out.reshape((-1, EMBEDD_DIM))
