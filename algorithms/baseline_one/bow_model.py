from torch import nn
import torch.nn.functional as F


class BoWClassifier(nn.Module):
    """
    Used to represent the layers our ANN will consist of together with the activation function.
    """
    def __init__(self, vocab_size, num_labels):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)
