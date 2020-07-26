from argparse import ArgumentParser
from collections import OrderedDict

import constants
import os


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

parser = ArgumentParser()
parser.add_argument("-d", choices=["test_data.txt", "train_pos_full.txt", "train_neg_full.txt", "train_pos.txt", "train_neg.txt"])
parser.add_argument("-s", type=str)
args = parser.parse_args()


text = ""
with open(os.path.join(constants.DATASETS_PATH, str(args.d)), "r", encoding='utf8') as f:
    text = f.read()

if args.s == "smile":
    for meaning in emoji_dictionary.keys():
        for (i, emoji) in enumerate(emoji_dictionary[meaning]):
            emoji = emoji.lower()
            spaced_emoji = ' '.join(list(emoji))
            text = text.replace(emoji, ' {} '.format(meaning))
            text = text.replace(spaced_emoji, ' {} '.format(meaning))


# Remove duplicates
text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))

if args.s == "smile":
    with open(os.path.join(constants.DATASETS_PATH, f"nodup-smileys-{args.d}"), "w", encoding='utf8') as outputfile:
        outputfile.write(text)
else:
    with open(os.path.join(constants.DATASETS_PATH, f"nodup-{args.d}"), "w", encoding='utf8') as outputfile:
        outputfile.write(text)
