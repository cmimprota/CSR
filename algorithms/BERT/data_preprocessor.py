import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

f = open("/home/xiaochenzheng/Desktop/cil-spring20-project-data_preprocessing/twitter-datasets/train_neg.txt")
tweets = f.readlines()
f.close()

# Incorrect stopwords
incorrect_stopwords = ['im', "shes", 'whats', 'thats', "dont", "arent", "couldnt", "didnt", "doesnt", "hadnt", "hasnt",
                       "havent", "isnt", "mightnt", "mustnt", "neednt", "shant", "shouldnt", "wasnt", "werent", "wont",
                       "wouldnt", 'cannot']

htmltag = re.compile('<.*?>')

for i, _tweet in enumerate(tweets):
    _tweet = re.sub(htmltag, ' ', _tweet) # remove <user><url>
    _tweet = re.sub('[^a-z A-Z0-9_#]', ' ', _tweet)  # Keep hashtag, remove punctuation
    _tweet = re.sub(' \d+', ' ', _tweet)  # Remove digits
    _tweet = [w for w in _tweet.split() if w not in stopwords.words('english') and w not in incorrect_stopwords and
              len(w) >= 3]
    tweets[i] = _tweet

wnl = WordNetLemmatizer()
for i, _tweet in enumerate(tweets):
    tidy = [wnl.lemmatize(w) for w in _tweet]
    tweets[i] = tidy

# Detokenized
for i, _tweet in enumerate(tweets):
    _tweet = ' '.join(_tweet)
    tweets[i] = _tweet