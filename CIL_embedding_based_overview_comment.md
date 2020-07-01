Embedding-based methods
=== 

This document and the corresponding drawing `CIL_embedding_based_overview.jpg` are meant to facilitate thinking about the overall pipeline from tweets to classification.

### Comments on the drawing
- The squares represent data that can be either computed on the fly (in-pipeline) or saved to file and reloaded as needed (out-of-pipeline)
	- This notion of pipeline is the same as in scikit-learn, the idea is enable automated hyperparameter grid-search
	- Of course even when computing in-pipeline, saving intermediate results to file can be useful in case of failure of later stages of the script
- The arrows u->v represent stages of the pipeline, i.e algorithms with u as input and v as output
- This drawing clearly captures glove-based methods, but also captures BOW: just think of the word-embedding as [word #i -> one-hot-encoding(i)] and the embedded tweets as the sum of the word embeddings
- Each square depends on (which raw data was used and) which algorithms/parameters were used in the preceding arrows, so there are many possible combinations. For reproducibility, all of that should be logged.
- In principle the algorithms can be chosen modularly, e.g a predictor can be applied on any embedded-tweets-data regardless of how it was obtained. However there are two stages that are not entirely modular:
	- even though a sentence embedding algorithm can be applied on any preprocessed tweets data using any word-embedding-map, if the vocabulary of the word-embedding-map doesn't match the preprocessed tweets data then we will encounter out-of-vocabulary words very often, which is not good.
	- In practice pre-built word embeddings are quite large, so it should be ok (modulo looooove issues, which can be dealt with either at the preprocessing stage or during out-of-vocabulary management). 
	- The other stage that is not completely modular is the word embedding. (If we use pre-built word embeddings then it's not a problem.) The reason is obvious: if the cut vocabulary doesn't come from the preprocessed tweets data, then it's pointless.
- If we choose to fine-tune a pre-built word embedding, then we must add a second input node to the graph (the initial word embedding).

### In the code

#### Structure at the root level
1. `./data_preprocessing/` contains code to preprocess raw tweets. 
2. `./vocab_building/` contains code to construct a vocabulary from (preprocessed) tweets.
3. `./word_embedd_building/` contains code to build a word-embedding from scratch, or fine-tune an existing word-embedding.
4. `./tweet_embedding`: contains code to (obtain a class that will) convert sentences to vectors
5. `./predicting_from_vect`: contains code to classify vectors

`./algorithms/embeddings_based`: contains code that puts everything together

#### Structure for each stage
All the directories described above follow essentially the same structure: each of them
- corresponds to a stage of the pipeline, 
- corresponds to a python package (hence the `__init__.py` everywhere)
- corresponds to a python class that is meant to be used for in-pipeline computations
- contains a "main file" that defines and exports that class; all the others files *implement* that class
- contains a `as_script.py` script that is meant to used for out-of-pipeline computations
- contains a `results` subdirectory to store intermediary results
	- it contains the output files when performing the stage as a script
	- *for some stages*, when used in-pipeline it also stores intermediary results there (think of it as a cache, useful in case of failure of later stages)

#### Typical usage
Since they follow the same structure, the directories have a similar usage.
See e.g `data_preprocessing/README.md` for an example; all other stages have essentially the same behaviour.

#### Compared to previous structure

Stevan's proposed structure was not really adapted to the idea of the multi-stage pipeline: it used a configuration log file, whose number of fields would have grown very large. Moreover I don't like having to rely on importing a global `constants.py` in all the python files, to know where to write output (and fetch input). 

Hence the new structure for embeddings-based methods. The new structure can probably be made more or less compatible with the previous one, but no attention was paid to this (sorry).
