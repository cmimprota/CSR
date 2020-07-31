This directory contains:
- the meanembedding and SIF baselines (written in jupyter notebook)
- WordCNN: a solution based on RNN+CNN (in that order), using the GloVe *pretrained* embedding as input
- CharResCNN_gw: same as Xiaochen's CharResCNN, just rewritten in a way that I find cleaner
- Char+WordCNN: an attempt to get a better performance by combining WordCNN and Char(Res)CNN

along with preprocessing scripts that are required to run them:
- preprocess-twitter.rb: slightly modified version of the Ruby script fetched on [GloVe's webpage](https://nlp.stanford.edu/projects/glove/), where we also downloaded the GloVe pretrained embedding
- preprocess_stanfordglove: a notebook that preprocesses using the "stanfordglove" method (see notebook for details)
- run_script.sh: a bash script to run the Ruby script, as an alternative to the `%%script ruby` cell in preprocess_stanfordglove.ipynb
- longest_tweet.py: a tiny utility script to determine the longest tweet length

The heavy data that need to be loaded are:
- twitter-datasets: copy the dataset available on kaggle
- (the glove embeddings cache, in CIL-aux-data, are downloaded automatically by `torchtext.vocab.GloVe` when running the notebooks)

The rest of the organization is straightforward and is easily guessed when running the scripts. In particular
- model checkpoints are stored in CIL-results/my_checkpoints
- submissions are stored in ./ (the "root directory")

Important: the notebooks all rely on the `ROOT_PATH` global variable, which is by default set to a path relative to my Google drive. That variable should be set to here, i.e the "root directory", i.e the directory that contains this README. For minimal risk of headache, I recommend setting it manually as needed, to the relevant *absolute path*.
