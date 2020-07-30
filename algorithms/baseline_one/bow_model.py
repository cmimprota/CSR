from torch import nn
import torch.nn.functional as F


class BoWClassifier(nn.Module):
    """
    Used to represent the layers our ANN will consist of together with the activation function.
    """
    def __init__(self, vocab_size, num_labels):
        super(BoWClassifier, self).__init__()
        # We have 2 layers only and no hidden layers
        # 1 input layer that has as many nodes as length of vocabulary and 1 output with 2 nodes
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        # Softmax function is applied on the 2 output nodes and it gives us log probabilities
        # So our output is vector of length two of log probabilities - one for each class
        # Input of the NLLLoss will define which position in vector we are optimizing and which class does it represent
        # We need to the exp it to get back the real probabilities that sum up to 1 and do prediction
        return F.log_softmax(self.linear(bow_vec), dim=1)


"""
# Another approach could be using BCELoss in combination with sigmoid in which case we have single node in output layer
# In that case negative class is for output < 0.5 and positive for > 0.5
# (if we put positive class in sigmoid as 0 then it is opposite)
"""
