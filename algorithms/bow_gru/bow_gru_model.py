import torch
from torch import nn
import torch.nn.functional as F


class BoWGRUClassifier(nn.Module):
    """
    Used to represent the layers our ANN will consist of together with the activation function.
    """
    def __init__(self, input_size, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BoWGRUClassifier, self).__init__()
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bow_gru_matrix):
        # bow_gru_matrix = [batch size, sent len, emb dim]
        _, hidden = self.rnn(bow_gru_matrix)
        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        # hidden = [batch size, hid dim]
        output = self.out(hidden)
        # output = [batch size, out dim]
        return output

