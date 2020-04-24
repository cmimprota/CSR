import os
import sys
import random
import pandas as pd
import numpy as np
from argparse import ArgumentParser

import torch
from torchtext import data
from transformers import BertTokenizer

sys.path.append('/home/xiaochenzheng/DLProjects/CIL/cil-spring20-project')
import constants

SEED = 2333

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

training_set = ArgumentParser(description='Training options for BERT model')
training_set.add_argument("--dataset", default="train-short", choices=["train-short", "train-full"], type=str,
                          help="Whether use full data or not")
training_set.add_argument("--dataset_file", default="dataset.csv", type=str,
                          help="Save txt as csv and its file name")
args = training_set.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_tweets_to_csv(file, label):
    if isinstance(label, int):
        assert label==0 or label==1, ("Label value of {} should be 1 for positive sentiment, 0 for negative sentiment".format(label))
    else:
        raise ValueError('Label must be int, got {}'.format(type(label)))
    # Load the TXT file of tweets
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        tweets = f.readlines()
    # Convert list to DataFrame with 'label' column
    tweets = pd.DataFrame(tweets, columns=['tweets'])
    tweets['label'] = label
    return tweets


if args.dataset == "train-short":
    parse_positive = "train_pos.txt"
    parse_negative = "train_neg.txt"
else:
    parse_positive = "train_pos_full.txt"
    parse_negative = "train_neg_full.txt"


tweets_pos = convert_tweets_to_csv(parse_positive, 1)
tweets_neg = convert_tweets_to_csv(parse_negative, 0)
tweets_dataset = tweets_pos.append(tweets_neg)
# Shuffle
tweets_dataset = tweets_dataset.sample(frac=1).reset_index(drop=True)
# Save CSV
tweets_dataset.to_csv(os.path.join(constants.DATASETS_PATH, args.dataset_file), index=False, sep=',')
print('Finish csv')


# Use BERT vocabulary
# https://github.com/bentrevett/pytorch-sentiment-analysis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens


# Define Field
TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)

fields = [('tweet', TEXT), ('label', LABEL)]

# Instantiation
tweets_data = data.TabularDataset(path=os.path.join(constants.DATASETS_PATH, args.dataset_file),
                                  format="CSV", fields=fields, skip_header=True)

# Validation set
train_data, val_data = tweets_data.split(split_ratio=0.8)
