import os
import sys
import torch
from torchtext import data
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import argparse
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithms.BERT.bert_model import BERTGRU
from algorithms.helpers import load_model, save_submission, load_vocabulary
import constants




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


training_set = ArgumentParser(description='Training options for BERT model')
training_set.add_argument("--dataset", default="train-short", choices=["train-short", "train-full"], type=str,
                          help="Whether use full data or not")
training_set.add_argument("--dataset_file", default="dataset.csv", type=str,
                          help="Save txt as csv and its file name")
training_set.add_argument("--train_file", default="train_set.csv", type=str,
                          help="Save training set file name")
training_set.add_argument("--valid_file", default="valid_set.csv", type=str,
                          help="Validation set file name")
training_set.add_argument("--pickle_path", default=os.path.join(constants.ROOT_DIR, "algorithms/BERT"), type=str,
                          help="Folder to save vocab.pkl ")
training_set.add_argument("--pickle_file", default="vocab.pkl", type=str,
                          help="pickle file name")
training_set.add_argument("--batch_size", default=128, type=int,
                          help="batch size")
training_set.add_argument("--hidden_dim", default=256, type=int,
                          help="Number of hidden dimensions for the Gated Recurrent Unit, the embedding dimension size")
training_set.add_argument("--output_dim", default=1, type=int,
                          help="Output dimension, for the sentiment task, it should be 1")
training_set.add_argument("--num_layers", default=2, type=int,
                          help="Number of recurrent layers")
training_set.add_argument("--bidirectional", default=True, type=str2bool,
                          help="Whether is a bidirectional model")
training_set.add_argument("--dropout_ratio", default=0.25, type=float)
training_set.add_argument("--num_epochs", default=50, type=int)
training_set.add_argument("--lr", default=0.01, type=float,
                          help="initial learning rate")
training_set.add_argument("--patience", default=5, type=int,
                          help="patience for early stopping")
training_set.add_argument("--gamma", default=0.1, type=float,
                          help='Gamma update for optimizer')
args = training_set.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens


def inference():
    with open(os.path.join(constants.DATASETS_PATH, "test_data_clean.txt"), "r") as f:
        tweets = f.readlines()
    # All tokens should be the index of vocabulary
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    # Define Field
    # TEXT = data.Field(batch_first=True,
    #                   use_vocab=False,
    #                   tokenize=tokenize_and_cut,
    #                   preprocessing=tokenizer.convert_tokens_to_ids,
    #                   init_token=init_token_idx,
    #                   eos_token=eos_token_idx,
    #                   pad_token=pad_token_idx,
    #                   unk_token=unk_token_idx)
    # fields = [('tweet', TEXT)]
    #
    # # Dataset and Iterator Instantiation
    # test_data = data.TabularDataset(path=os.path.join(constants.DATASETS_PATH, "test_data.txt"),
    #                                   format="CSV", fields=fields, skip_header=True)
    #
    # test_iterator = data.BucketIterator.splits(test_data, sort=False, batch_size=args.batch_size, device=device)

    # Model Instantiation
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERTGRU(bert=bert,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.output_dim,
                    n_layers=args.num_layers,
                    bidirectional=args.bidirectional,
                    dropout=args.dropout_ratio)
    model = model.to(device)
    model.load_state_dict(torch.load('/home/xiaochenzheng/DLProjects/CIL/cil-spring20-project/algorithms/BERT/weights/42_model.pt', map_location=device))
    model.eval()

    label_predictions = []
    with tqdm(total=len(tweets), desc="Inference", leave=False) as pbar:
        with torch.no_grad():
            for i, tweet in enumerate(tweets):


                tokens = tokenizer.tokenize(tweet)
                tokens = tokens[:max_input_length - 2]
                indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(0)
                predictions = torch.sigmoid(model(tensor))
                label_predictions.append(1 if predictions > 0.5 else -1)
                pbar.update()

    save_submission(label_predictions, "/home/xiaochenzheng/DLProjects/CIL/cil-spring20-project/algorithms/BERT/results")

if __name__ == "__main__":
    inference()