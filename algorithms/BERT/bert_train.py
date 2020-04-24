import os
import sys
import pandas as pd

import torch
from torchtext import data
from transformers import BertTokenizer

sys.path.append('/home/xiaochenzheng/DLProjects/CIL/cil-spring20-project')
import constants

dataset_file = 'dataset.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_tweets_to_csv(file_neg, file_pos, output_file=dataset_file):
    # Load the TXT file of tweets
    with open(os.path.join(constants.DATASETS_PATH, file_neg), "r") as f:
        tweets_neg = f.readlines()
    with open(os.path.join(constants.DATASETS_PATH, file_pos), "r") as f:
        tweets_pos = f.readlines()
    # Convert list to DataFrame with 'label' column
    tweets_pos = pd.DataFrame(tweets_pos, columns=['tweets'])
    tweets_pos['label'] = 'pos'
    tweets_neg = pd.DataFrame(tweets_neg, columns=['tweets'])
    tweets_neg['label'] = 'neg'
    tweets_dataset = tweets_pos.append(tweets_neg)
    # Shuffle
    tweets_dataset = tweets_dataset.sample(frac=1).reset_index(drop=True)
    # Save CSV
    tweets_dataset.to_csv(os.path.join(constants.DATASETS_PATH, output_file), index=False, sep=',')


convert_tweets_to_csv('train_neg_full.txt', 'train_pos_full.txt')

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
tweets_train = data.TabularDataset(path=os.path.join(constants.DATASETS_PATH, dataset_file),
                                   format="CSV", fields=fields, skip_header=True)


