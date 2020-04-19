from torch import nn
import torch.nn.functional as F

# Could not import with - in the naming convention


class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)
