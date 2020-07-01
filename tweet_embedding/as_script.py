import os
import pickle
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

RESULTS_SUBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from .tweet_embedder import *

def main():
    parser = ArgumentParser()
    parser.add_argument("--vocab", type=str, required=True, help="vocabulary (full path to txt file)")
    parser.add_argument("--word-embedd", "--we", required=True, help="word embedding to use (full path to npz file)")
    parser.add_argument("--method", "-m" choices=["mean", "raw"], default="mean")
    parser.add_argument("--pad-mode", type=str) # TODO: also allow floats
    parser.add_argument("--max-tweet-length", type=int)
    parser.add_argument("to_embed", help="(preprocessed) tweets to embed (full path to txt file)")
    args = parser.parse_args()

    import warnings
    if args.method == "raw":
        if args.pad_mode is None:
            warnings.warn("parameter args.epochs is required when using fresh-glove method")
        if args.max_tweet_length is None:
            warnings.warn("parameter args.embedd_dim is required when using fresh-glove method")
    
    if args.output is not None:
        outdir = os.path.dirname(args.output)
        outfile = args.output
    else:
        import time
        pretty_name = f"tweet_embeddings__{time.time()}" # just the timestamp... 
        # TODO: maybe also keep a log of what method was used somewhere (otw it's OK, don't need anything more to do the prediction from here)
        print(f"inferred pretty name: {pretty_name}")
        outdir = os.path.join(RESULTS_SUBDIR, pretty_name)

        if args.method == "mean":
            outfilename = args.method
        elif args.method == "raw":
            outfilename = f"{args.method}__pad{args.pad}__mtl{args.max_tweet_length}"
        else:
            raise ValueError("unrecognized tweet-embedding method")
        outfile = os.path.join(outdir, outfilename)

    # create output dir if it doesn't exist yet
    os.makedirs(outdir, exist_ok=True)

    with open(args.vocabulary) as vocabfile:
        vocab = (word.rstrip() for word in vocabfile)
    with open(args.tweets_to_embed) as tweetsfile:
        tweets = (tweet.rstrip() for tweet in tweetsfile) # (these are preprocessed tweets)
    with np.load(word_embedd_fn) as data: # TODO: check whether this syntax works for .npy as well
        word_embedd = data
        # xs = data['xs'] # (for .npz)

    TE = create_TweetEmbedder(vocab=args.vocab, word_embedd=word_embedd, method=args.method, **args)

    res = np.empty_like(len(tweets), TE.OUTPUT_DIM)
    for i, tweet in tqdm(enumerate(tweets)):
        res[i] = TE.embed_tweet(tweet)

    # np.savez(f"{outfile}.npz"), xs=xs, ys=ys)
    np.save(f"{outfile}.npy"), word_embedding)

if __name__ == '__main__':
    main()
