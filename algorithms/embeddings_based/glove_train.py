# INPUT: -v <vocabulary filename>
#        -d <dataset filename> (defaults to short dataset)

# OUTPUT: pickled sklearn model

import os
import pickle
from argparse import ArgumentParser
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ) # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import constants
from algorithms.fresh_glove_embedding.tweet_embedder import TweetEmbedder, create_TweetEmbedder

parser = ArgumentParser()
parser.add_argument("-v", type=str, required=True, help="vocabulary (without path, without extension)")
parser.add_argument("-emb-d", choices=["train-short", "train-full"], default="train-short", help="training dataset used in GLOVE (to choose which embedding to use)")
parser.add_argument("-emb-meth", choices=["mean", "raw"], default="mean", help="tweet-embedding method used in GLOVE (to choose which embedding to use)")
parser.add_argument("pos_training_set", choices=["train_pos_full", "train_pos"])
parser.add_argument("neg_training_set", choices=["train_neg_full", "train_neg"])
parser.add_argument("--do-holdout", action="store_true", help="completely put aside a part of the training set (X_mytest) to sanity check against overfitting") # default to False
args = parser.parse_args()

# assuming everything was embedded as a previous step, don't even need an EmbedTweets instance at all
# ET = create_EmbedTweets(vocab_fn=args.v, dataset_fn=args.emb_d, method=args.emb_meth)

embedded_datasets_dir = os.path.join(constants.ROOT_DIR, "algorithms", "fresh_glove_embedding", f"embedded-datasets__{args.emb_d}__{args.v}__{args.emb_meth}")
X_pos = np.load(os.path.join(embedded_datasets_dir, f"embedded_{args.pos_training_set}.npy"))
X_neg = np.load(os.path.join(embedded_datasets_dir, f"embedded_{args.neg_training_set}.npy"))
y_pos = np.ones(X_pos.shape[0])
y_neg = np.zeros(X_neg.shape[0])
X_train = np.concatenate((X_pos, X_neg))
y_train = np.concatenate((y_pos, y_neg))

if (args.do_holdout):
    X_train, X_mytest, y_train, y_mytest = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
    print("read *_train.csv, split into len(X_train)=len(y_train)=", len(X_train), 
        ", len(X_mytest)=len(y_mytest)=", len(X_mytest))
else:
    X_mytest = X_train[0:2]
    y_mytest = y_train[0:2]
    print("read *_train.csv, len(X_train)=len(y_train)=", len(X_train), 
        " (and put dummy data in X_mytest, y_mytest)")

# count occurence of each class in the X/y_train set, to check that dataset is balanced
print("Class 0:", np.count_nonzero(y_train==0), "\tIn proportion of dataset:", np.count_nonzero(y_train==0)/len(y_train))
print("Class 1:", np.count_nonzero(y_train==1), "\tIn proportion of dataset:", np.count_nonzero(y_train==1)/len(y_train))

# define the pipeline
# -------------------
print("definining the transformers and classifier...")
# Transformers e.g:
# standardize (center and scale) data https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
# feature selection at pre-processing https://scikit-learn.org/stable/modules/feature_selection.html

# binary classifier estimator https://scikit-learn.org/stable/modules/svm.html#classification
clf = LinearSVC() # the rest of the params is grid-searched

# optional further step: use ensemble methods https://scikit-learn.org/stable/modules/ensemble.html#bagging

pipe = Pipeline([
    ('standardize', StandardScaler()),
    # ('vt_feat_select', VarianceThreshold()),
    ('feat_select', SelectPercentile(score_func=f_classif)),
    ('classification', clf),
])
print("defined pipeline `pipe`.")
# Parameters available to grid-search:
# pipe.get_params().keys()
# clf.get_params().keys()

# train and evaluate the model
# ----------------------------
# CV-train
param_grid = {
    'feat_select__percentile': (100,), # (80, 90, 100,),
    'classification__C': (0.1, 0.4, 1), 
    # 'classification__coef0': (0, 1.0),
    # 'classification__degree': (2, 3, 4), # only for kernel='poly'
    # 'classification__gamma': ('auto', 'scale'),
    # 'classification__kernel': ('rbf', 'linear', 'sigmoid', 'poly'),
    # 'classification__kernel': ('rbf',),
}
print("CV-training pipe on X_train,y_train (using grid search)...\n")
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=4, verbose=1)
grid_search.fit(X_train, y_train)
print("\ntrained pipe into `grid_search`.")

# save model
PATH_FOR_TRAINED_MODEL = os.curdir # TODO: find somewhere to put it
joblib.dump(gridsearch, os.path.join(PATH_FOR_TRAINED_MODEL, "trained_model.pkl"))
print("saved model to " + os.path.join(PATH_FOR_TRAINED_MODEL, "trained_model.pkl"))

# show results
print(f"Best score: {grid_search.best_score_}")
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in param_grid.keys():
    # print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print(f"\t{param_name}, {best_parameters[param_name]}")

do_show_detailed = True
if (do_show_detailed):
    print("\nGrid scores on development set:")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        # print("%0.3f (+/-%0.03f) for %r"
        #       % (mean, std * 2, params))
        print(f"{mean} (+/-{std*2} for {params}")

# predict
y_predtrain = grid_search.predict(X_train)
y_mypred = grid_search.predict(X_mytest)

# evaluate
score_train = accuracy_score(y_train, y_predtrain)
print(f"(Unvalidated) accuracy score on the training set: {score_train}")
score_mytest = balanced_accuracy_score(y_mytest, y_mypred)
print(f"accuracy score on the `mytest` set: {score_mytest}")

print("confusion matrix for training set:")
print(confusion_matrix(y_train, y_predtrain))
print("confusion matrix for `mytest` set:")
print(confusion_matrix(y_mytest, y_mypred))
