import os
import sys
import random
import pandas as pd
import numpy as np
import argparse
from argparse import ArgumentParser
from tqdm import tqdm
import math
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from transformers import BertTokenizer, BertModel

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithms.BERT.bert_model import BERTGRU
import constants

SEED = 2333
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


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


def convert_tweets_to_csv(file, label):
    if isinstance(label, int):
        assert label==-1 or label==1, ("Label value of {} should be 1 for positive sentiment, -1 for negative sentiment".format(label))
    else:
        raise ValueError('Label must be int, got {}'.format(type(label)))
    # Load the TXT file of tweets
    with open(os.path.join(constants.DATASETS_PATH, file), "r") as f:
        tweets = f.readlines()
    # Convert list to DataFrame with 'label' column
    tweets = pd.DataFrame(tweets, columns=['tweets'])
    tweets['label'] = label
    return tweets


def csv_split(train_ratio):
    # Generating CSV for further instantiation (class torchtext.data.Dataset)
    if args.dataset == "train-short":
        parse_positive = "train_pos.txt"
        parse_negative = "train_neg.txt"
    else:
        parse_positive = "train_pos_full.txt"
        parse_negative = "train_neg_full.txt"
    tweets_pos = convert_tweets_to_csv(parse_positive, 1)
    tweets_neg = convert_tweets_to_csv(parse_negative, -1)
    tweets_dataset = tweets_pos.append(tweets_neg)
    # Shuffle
    tweets_dataset = tweets_dataset.sample(frac=1).reset_index(drop=True)
    # Splitting
    N = len(tweets_dataset)
    train_len = int(round(train_ratio * N))
    # Save CSV
    if train_ratio == 1:
        tweets_dataset.to_csv(os.path.join(constants.DATASETS_PATH, args.train_file), index=False, sep=',')
    else:
        tweets_dataset[:train_len].to_csv(os.path.join(constants.DATASETS_PATH, args.train_file), index=False, sep=',')
        tweets_dataset[train_len:].to_csv(os.path.join(constants.DATASETS_PATH, args.valid_file), index=False, sep=',')
    tweets_dataset.to_csv(os.path.join(constants.DATASETS_PATH, args.dataset_file), index=False, sep=',')


csv_split(0.8)
# Use BERT vocabulary
# https://github.com/bentrevett/pytorch-sentiment-analysis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Define Binary accuracy
# https://github.com/bentrevett/pytorch-sentiment-analysis
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# Define lr adjustment strategy
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pickle_vocab(vocab):
    """
    Args:
        vocab: List or Dict of vocabulary
    """
    with open(os.path.join(args.pickle_path, args.pickle_file), 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    return args.pickle_file


def train_process():
    # All tokens should be the index of vocabulary
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    # Save vocab
    vocab = tokenizer.vocab
    pickle_vocab(vocab)

    # Define Field
    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)
    LABEL = data.LabelField(dtype=torch.float)
    fields = [('tweet', TEXT), ('label', LABEL)]

    # Dataset and Iterator Instantiation
    tweets_data = data.TabularDataset(path=os.path.join(constants.DATASETS_PATH, args.dataset_file),
                                      format="CSV", fields=fields, skip_header=True)
    train_data, valid_data = tweets_data.split(split_ratio=0.8, random_state=random.seed(SEED))
    # TODO Two methods to split train-valid 1)by sdf csv_split; 2)by data.Dataset.split(random_state=random.seed(SEED)) (but we can not save the train-valid dataset)
    # train_data, valid_data = data.TabularDataset.splits(path=constants.DATASETS_PATH, train=args.train_file,
    #                                                     validation=args.valid_file, format='csv',
    #                                                     fields=fields, skip_header=True)
    print("-"*60+"\nThere are {} training examples and {} validation examples.".format(len(train_data), len(valid_data)))
    LABEL.build_vocab(train_data)
    print("-"*60+"\nThe vocab of labels is {}".format(LABEL.vocab.stoi))
    train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data),
                                                                sort=False, batch_size=args.batch_size, device=device)

    # Model Instantiation
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERTGRU(bert=bert,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.output_dim,
                    n_layers=args.num_layers,
                    bidirectional=args.bidirectional,
                    dropout=args.dropout_ratio)

    # Set the BERT parameters un-trainable.
    # https://github.com/bentrevett/pytorch-sentiment-analysis
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("\nThe trainable parameters are parameters of the GRU (rnn) and the linear layer (out):"+"-"*60)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("-"*60)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Because the label is either 0 or 1. Sigmoid followed by a BCELoss. But the output of model is a 1-dim scalar, no sigmoid.
    #TODO For the submission, use preds = model(batch.tweet).squeeze(1), submission = 2*torch.sigmoid(preds) - 1 or tanh???
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # Training process
    best_valid_loss = float('inf')
    valid_epoch_loss_list = np.empty(args.num_epochs)
    num_iter_per_train_epoch = math.ceil(len(train_data) / args.batch_size)
    num_iter_per_valid_epoch = math.ceil(len(valid_data) / args.batch_size)

    step_index = 0
    patience_count = 0

    for epoch in tqdm(range(args.num_epochs), desc="Epochs", position=0):

        with tqdm(total=num_iter_per_train_epoch, desc="Training", leave=False) as pbar:
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for iteration, batch in enumerate(train_iterator):
                optimizer.zero_grad()
                predictions = model(batch.tweet).squeeze(1)
                loss = criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
                train_epoch_acc += acc.item()

                pbar.set_postfix(Sample_Loss=loss.item(), Binary_Accuracy=acc.item())
                pbar.update()

        with tqdm(total=num_iter_per_valid_epoch, desc="Validation", leave=False) as pbar:
            valid_epoch_loss = 0
            valid_epoch_acc = 0
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(valid_iterator):
                    predictions = model(batch.tweet).squeeze(1)
                    loss = criterion(predictions, batch.label)
                    acc = binary_accuracy(predictions, batch.label)
                    valid_epoch_loss += loss.item()
                    valid_epoch_acc += acc.item()
                    valid_loss = valid_epoch_loss / (iteration+1)

                    pbar.set_postfix(Sample_Loss=loss.item(), Binary_Accuracy=acc.item())
                    pbar.update()

        valid_epoch_loss_list[epoch] = valid_loss
        print('-'*60+'\nEpoch: {}'.format(epoch+1))
        # print('the learning rate is {}'.format(args.lr))
        print('the average loss of validation data is {}'.format(valid_loss))

        if (best_valid_loss - valid_epoch_loss_list[epoch]) > 0.000001: # valid_epoch_loss_list[epoch + 1] < best_valid_loss

            best_valid_loss = valid_epoch_loss_list[epoch]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "weights/"+str(best_epoch)+"_model.pt")
            print('Saving model at epoch: {}'.format(best_epoch))  # Save periodically
            patience_count = 0
        else:
            patience_count += 1

        if patience_count > args.patience:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            print("-"*60+"\nearly stopping...change the learning rate to {}".format(args.lr * (args.gamma ** step_index)))
            patience_count = 0


if __name__ == '__main__':
    train_process()
