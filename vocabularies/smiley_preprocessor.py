from argparse import ArgumentParser
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
parser.add_argument("-d", choices=["test", "train-short", "train-full", "test-and-train-short", "test-and-train-full"])
args = parser.parse_args()

files_to_parse = []

if args.d == "test":
    files_to_parse.append("test_data.txt")
elif args.d == "train-short":
    files_to_parse.append("train_pos.txt")
    files_to_parse.append("train_neg.txt")
elif args.d == "train-full":
    files_to_parse.append("train_pos_full.txt")
    files_to_parse.append("train_neg_full.txt")
elif args.d == "test-and-train-short":
    files_to_parse.append("train_pos.txt")
    files_to_parse.append("train_neg.txt")
    files_to_parse.append("test_data.txt")
elif args.d == "test-and-train-full":
    files_to_parse.append("train_pos_full.txt")
    files_to_parse.append("train_neg_full.txt")
    files_to_parse.append("test_data.txt")

for file in files_to_parse:
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        text = f.read()

for meaning in emoji_dictionary.keys():
    for (i, emoji) in enumerate(emoji_dictionary[meaning]):
        emoji = emoji.lower()
        spaced_emoji = ' '.join(list(emoji))
        text = text.replace(emoji, ' {} '.format(meaning))
        text = text.replace(spaced_emoji, ' {} '.format(meaning))

with open(os.path.join(constants.DATASETS_PATH, f"emoji-{args.d}.txt"), "w") as outputfile:
    outputfile.write(text)
