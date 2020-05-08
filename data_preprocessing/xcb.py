import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from data_preprocessor import DataPreprocessor

class XcbDP(DataPreprocessor):
    """
    Preprocessing tweets using the xcb method (aka the script that was initially used by *X*iao*C*hen for *B*ERT)
    """
    def __init__(self):
        super().__init__()
        self.incorrect_stopwords = ['im', "shes", 'whats', 'thats', "dont", "arent", "couldnt", "didnt", "doesnt", "hadnt", "hasnt",
                        "havent", "isnt", "mightnt", "mustnt", "neednt", "shant", "shouldnt", "wasnt", "werent", "wont",
                        "wouldnt", 'cannot']
        self.htmltag = re.compile('<.*?>')
        self.wnl = WordNetLemmatizer()

    def preprocess(self, tweet):
        """
        Args:
            tweet (str): the raw tweet
        Returns:
            preprocessed_tweet (str): the preprocessed tweet as a string
        """
        tweet = re.sub(self.htmltag, ' ', tweet)        # Remove <user><url>
        tweet = re.sub('[^a-z A-Z0-9_#]', ' ', tweet)   # Keep hashtag, remove punctuation
        tweet = re.sub(' \d+', ' ', tweet)              # Remove digits
        tweet = [w for w in tweet.split() if w not in stopwords.words('english') and w not in self.incorrect_stopwords and len(w) >= 3]
        
        tweet = [self.wnl.lemmatize(w) for w in tweet]       # Tidy

        tweet = ' '.join(tweet)                         # Detokenized

        return tweet