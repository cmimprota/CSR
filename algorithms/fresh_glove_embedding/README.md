> Rk: this was useful to get a baseline at first; in practice it might be easier to use pytorch's pretrained embeddings.
> If there are no pre-coded pytorch methods for fine-tuning of embedding then this can still be useful...

For the GLOVE model, see the slides for Lecture 5 (Word embeddings) Section 4, i.e pp. 17-26

See Exercise 6 Problems 5, 6 and [the correction](https://github.com/dalab/lecture_cil_public/tree/master/exercises/ex6), which this is based on

### constructing a GLOVE embedding
- use `cooc.py` to construct the co-occurrence matrix `cooc__<textdata>__<vocab>.pkl` from a specified vocabulary and the original twitter-datasets (full or short)
- use `glove_solution.py` to construct the embedding `embeddings__<textdata>__<vocab>.npz` from the corresponding co-occurrence matrix

### output format + how to use it
- this npz file can be loaded using [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load), namely
```python
with np.load(f'embeddings__{textdata}__{vocab}.npz') as data:
    xs = data['xs']
    ys = data['ys']
```
- `xs` and `ys` have the same shape, namely `(vocab_size, embedding_dim)`. So, each `xs[i,:]` corresponds to a word in the vocabulary
- recall that GLOVE model is: $log p = <xs[i], ys[j]>$ measures how often does word $i$ occur in a context of word $j$

### making tweet-embeddings
One way to use the embeddings for tweets classification is to also embed each tweet, i.e make a tweet-embedding based on the word-embedding. After that, can use any classical model for vector classification (linear or not, actually). There are multiple methods to make such an embedding.

The script `tweet_embedder.py` allows to
- embed all the tweets from a twitter-dataset, into file `embedded-datasets__<trainingdata>__<vocab>__<method>/embedded_<textdata>.npy` which can then be loaded by
```python
embedded_tweets = np.load(f'embedded_{textdata}')
# embedded_tweets[k,:] is the embedding of the k-th tweet
```
- through the exported class `TweetEmbedder`, a tweet from the test dataset can be embedded by calling `TE.embed(tweet)`


### extensions/TODOs
- looks like the provided solution didn't include separate bias terms, as described in Lecture 5 p.23...
- the provided solution defines the "context" of a word to be the whole tweet, whereas we could for example limit it to a few positions before and after the word. But it also makes sense to take the whole tweet, so okay.
