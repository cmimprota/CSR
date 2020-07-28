from argparse import ArgumentParser
from collections import OrderedDict
import constants
import os
import re
import nltk
from nltk.corpus import stopwords
import warnings


parser = ArgumentParser()
parser.add_argument("-d", choices=["test_data.txt", "train_pos_full.txt", "train_neg_full.txt", "train_pos.txt",
                                   "train_neg.txt"])
parser.add_argument("-s", type=str)
parser.add_argument("-n", type=str)
parser.add_argument("-p", type=str)
args = parser.parse_args()


output_filename = ""
text = ""
with open(os.path.join(constants.DATASETS_PATH, str(args.d)), "r", encoding='utf8') as f:
    text = f.read()


############################### D U P L I C A T E S     B E G I N ###############################


if args.n == "nodup":
    output_filename += "nodup-"
    text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))


############################### D U P L I C A T E S    E N D ###############################


############################### S M I L E Y     B E G I N ###############################


emoji_dictionary = {
    'happy': [':‑)', ':)', ':-]', ':]', ':-3', ':3', ':->', ':>', '8-)', '8)', ':-}', ':}', ':o)', ':c)', ':^)', '^.^',
              '=]', '(:', '=)', ':-))', '(y)'],
    'laughing': [':‑D', ':D', '8‑D', '8D', 'x‑D', 'xD', 'X‑D', 'XD', '=D', '=3', 'B^D'],
    'sad': [':‑(', ':(', ':‑c', ':c', ':‑<', ':<', ':‑[', ':[', ':-||', '>:[', ':{', ':@', ';(', '=(', '(n)'],
    'crying': [":'‑(", ":'("],
    'thrill': [";')", ":')"],
    'horror': ["D‑':", 'D:<', 'D:', 'D8', '	D;', 'D=', 'DX'],
    'surprise': [':‑O', ':O', ':‑o', ':o', ':-0', '8‑0', '>:O'],
    'love': ['<3'],
    'kiss': [':-*', ':*', ':×'],
    'wink': [';‑)', ';)', '*-)', '*)', ';‑]', ';]', ';^)', ':‑,', '	;D'],
    'playful': [':‑P', ';-)' ':P', 'X‑P', 'XP', 'x‑p', 'xp', ':‑p', ':p', ':‑Þ', ':Þ', ':‑þ', ':þ', '	:‑b', ':b',
                'd:', '=p', '>:P'],
    'annoyed': ['.__.', '._.', '-_-', '-__-', ':‑/', ':/', ':‑.', '>:\\', '>:/', ':\\', '	=/', '=\\', ':L', '=L',
                ':S'],
    'indecision': [':‑|', ':|'],
    'embarrassed': [':$', '://)', '://3'],
    'innocent': ['O:‑)', 'O:)', '0:‑3', '0:3', '0:‑)', '0:)', '0;^)'],
    'devilish': ['>:‑)', '}:‑)', '}:)', '3:‑)', '3:)', '>;)', '>:3', ';3'],
}

if args.s == "smileys":
    output_filename += "smileys-"
    for meaning in emoji_dictionary.keys():
        for (i, emoji) in enumerate(emoji_dictionary[meaning]):
            emoji = emoji.lower()
            spaced_emoji = ' '.join(list(emoji))
            text = text.replace(emoji, ' {} '.format(meaning))
            text = text.replace(spaced_emoji, ' {} '.format(meaning))


############################### S M I L E Y     E N D ###############################


############################### P R E P R O C E S S     B E G I N ###############################


if args.p == "preprocess":
    output_filename += "preprocess-"
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Incorrect stopwords
    incorrect_stopwords = ['im', "shes", 'whats', 'thats', "dont", "arent", "couldnt", "didnt", "doesnt", "hadnt", "hasnt",
                           "havent", "isnt", "mightnt", "mustnt", "neednt", "shant", "shouldnt", "wasnt", "werent", "wont",
                           "wouldnt", 'cannot']

    htmltag = re.compile('<.*?>')
    final_text = ""
    cachedStopWords = stopwords.words("english")
    for line in text.splitlines():
        line = re.sub(htmltag, ' ', line)  # remove <user><url>
        line = re.sub('[^a-z A-Z0-9_#]', ' ', line)  # Keep hashtag, remove punctuation
        line = re.sub(' \d+', ' ', line)  # Remove digits
        # text = '\n'.join([line for line in text.splitlines()])
        line = ' '.join([w for w in line.split() if w not in cachedStopWords and w not in incorrect_stopwords])
        final_text += line + '\n'

    #text = ' '.join([w for w in text.split() if w not in cachedStopWords and w not in incorrect_stopwords])


############################### P R E P R O C E S S     E N D ###############################


if args.s == "smileys":
    with open(os.path.join(constants.DATASETS_PATH, f"{output_filename}{args.d}"), "w", encoding='utf8') as outputfile:
        outputfile.write(final_text)
