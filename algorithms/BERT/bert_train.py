import pandas as pd
import csv

import torch
from torchtext import data
from transformers import BertTokenizer

PATH = "/home/xiaochenzheng/Desktop/cil-spring20-project-data_preprocessing/twitter-datasets/" # Dataset folder
FILE = "dataset.csv"

# Load the TXT file of tweets
f = open(PATH + "train_neg.txt")
tweets_neg = f.readlines()
f.close()

f = open(PATH + "train_pos.txt")
tweets_pos = f.readlines()
f.close()

# Convert list to DataFrame with 'label' column
tweets_pos = pd.DataFrame(tweets_pos, columns=['tweets'])
tweets_pos['label'] = 'pos'
tweets_neg = pd.DataFrame(tweets_neg, columns=['tweets'])
tweets_neg['label'] = 'neg'

tweet_dataset = tweets_pos.append(tweets_neg)

# Shuffle
tweet_dataset = tweet_dataset.sample(frac=1).reset_index(drop=True)

# Save CSV
tweet_dataset.to_csv(PATH+FILE, index=False)

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
tweets_train = data.TabularDataset(path=PATH+FILE, format="CSV", fields=fields, skip_header=True)


