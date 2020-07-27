import os
import pickle
import random
import csv
from argparse import ArgumentParser
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import device

import constants
from algorithms.baseline_one.bow_model import BoWClassifier
from algorithms.bow_gru.bow_gru_model import BoWGRUClassifier
from algorithms.helpers import save_model

# Same device needs to be used when instantiating tensors as for model. Exception thrown otherwise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is the maximum number of words that we will take into consideration from a single tweet
MAX_NO_OF_WORDS = 50


class TweetsDataset(Dataset):
    def __init__(self, label_data_path, vocab_path, l0=1000):
        """
        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            vocab_path: The path of vocab pickle file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        # read vocab
        self.loadVocab(vocab_path)
        self.load(label_data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadVocab(self, vocab_path):
        self.vocab = []
        with open(os.path.join(constants.VOCABULARIES_CUT_PATH, vocab_path), "rb") as f:
            ws = pickle.load(f)
            for w in ws:
                self.vocab.append(''.join(w))

    def load(self, label_data_path, lowercase=True):
        self.label = []
        self.data = []
        with open(os.path.join(constants.DATASETS_PATH, label_data_path), 'r', encoding='utf8') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                if index > 0:
                    self.label.append(int(row[1]))
                    txt = ' '.join(row[0:])
                    if lowercase:
                        txt = txt.lower()
                    self.data.append(txt)

        self.y = torch.FloatTensor(self.label, device=DEVICE)

    def oneHotEncode(self, idx):
        # X = (batch, vocab_length, tweet_length_of_words)
        X = torch.zeros(len(self.vocab), self.l0, device=DEVICE)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence.split()):
            if char in self.vocab:
                X[self.word2Index(char)][index_char] = 1.0
        return X

    def word2Index(self, character):
        return self.vocab.index(character)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", type=str, help="dataset - choose train-short or train-full")
    parser.add_argument("-v", type=str, help="vocabulary - filename in the folder cut")
    args = parser.parse_args()

    label_data_path = str(args.d)  # 'CIL_clean/dataset_clean.csv'
    vocab_path = str(args.v)  # 'vocabularies/cut/cut-vocab-test-and-train-full-1000-most-frequent.pkl'

    train_dataset = TweetsDataset(label_data_path, vocab_path)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, drop_last=False)

    # Read the model description in bow_gru_model.py
    model = BoWGRUClassifier(input_size=len(train_dataset.vocab),
                             hidden_dim=256,
                             output_dim=1,
                             n_layers=2,
                             bidirectional=True,
                             dropout=0.25)

    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    loss_function = nn.BCEWithLogitsLoss()
    loss_function = loss_function.to(DEVICE)

    losses = []

    for epoch in range(3):

        print(F"Running epoch {epoch}\n")

        # Accumulates the losses of the current epoch
        total_loss = 0
        print(F"1\n")
        for i, data in enumerate(train_loader):
            #local_batch, local_labels = local_batch.to(DEVICE), local_labels.to(DEVICE)
            local_batch, local_labels = data
            model.zero_grad()
            prediction = model(local_batch).squeeze(1)
            loss = loss_function(prediction, local_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_dataset))

    # Print losses of all epochs
    print(losses)

