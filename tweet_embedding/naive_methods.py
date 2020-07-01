from tweet_embedder import TweetEmbedder

class MeanVectorTE(TweetEmbedder): 
    """Represent the tweet by the mean of its word embeddings"""
    def __init__(self, vocab, word_embedd):
        super().__init__(vocab=vocab, word_embedd=word_embedd)

    def embed_tweet(self, tweet):
        l = [ self.xys[self.word_to_index[word]] if word in self.vocab else self.oov_vector for word in tweet ]
        return np.mean(np.array(l), axis=0)

class PaddedConcatTE(TweetEmbedder):
    """Represent the tweet by the concatenation of its word embeddings ("raw" embedding)
    - If tweet is longer than max_tweet_length, truncate it
    - If tweet is shorter than max_tweet_length, pad with pad_mode (by default, 0's)
    Rk: this method can be really heavy and is mainly there for convenience and testing
    
    Additionaly params:
        max_tweet_length (int): the assumed max length of the tweets (in number of words)
        pad_mode (str or function or float): the padding mode for https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            if float, pad with constant value (over all dimensions of the embedding)
    """
    def __init__(self, vocab, word_embedd, max_tweet_length=30, pad_mode=0.):
        super().__init__(vocab=vocab, word_embedd=word_embedd)
        self.max_tweet_length = max_tweet_length
        self.pad_mode = pad_mode

    def embed_tweet(self, tweet):
        l = [ self.xys[self.word_to_index[word]] if word in self.vocab else self.oov_vector for word in tweet ]
        concated = np.concatenate(np.array(l), axis=1)
        if len(tweet) < self.max_tweet_length:
            diff = self.EMBEDDING_DIM*( self.max_tweet_length-len(tweet) )
            if isinstance(self.pad_mode, str):
                concated = np.pad(concated, (0,diff), mode=self.pad_mode)
            elif isinstance(self.pad_mode, (int, float)):
                concated = np.pad(concated, (0,diff), constant_values=self.pad_mode)
            else: 
                raise ValueError("invalid `pad_mode` parameter")
        assert concated.shape == (self.max_tweet_length*self.EMBEDDING_DIM,)
        return concated
