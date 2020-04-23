# DATA PREPROCESSING
1. Remove meaningless content (HTML tag, punctuation, digits...) and do lemmatization. 
Script available [here](https://gitlab.ethz.ch/mstevan/cil-spring20-project/-/blob/master/algorithms/BERT/data_preprocessor.py).
2. Tokenize tweets using BERT pre-trained model which means the actual vocabulary from BERT pre-trained model will be used 
(documentation [here](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer)). Script available [here](https://gitlab.ethz.ch/mstevan/cil-spring20-project/-/blob/master/algorithms/BERT/bert_train.py).
3. Every token will be converted to their index of the vocabulary.

# DATASET INSTANTIATION
**Dataset format**
* Class [*torchtext.data.TabularDataset*](https://pytorch.org/text/data.html#tabulardataset) only supports three data formats: JSON, TSV, CSV.
* Class [*torchtext.data.Dataset*](https://pytorch.org/text/data.html#dataset) can only create a dataset from a python list.
* CSV will be used.

**Fields**
* Hard to explain, see [here](https://pytorch.org/text/data.html#fields)

**Validation set?**\
tbd

