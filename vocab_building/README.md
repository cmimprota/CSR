## Usage

Only use `vocab_builder.py`; all the others are auxiliary files.

The script `vocab_builder.py` allows to build a (cut) vocabulary from a preprocessed tweets dataset, into file `vocabularies/<preproc_method>__<base_data>/<preproc_meth>.txt`. The vocabulary is also stored in `<preproc_meth>.pkl` of the same directory, where it is represented as a (duplicate-free) list

The code follows the same structure as other stages of the pipeline, e.g `tweet_embedder`. As a free benefit of the code structure, through the exported class `VocabBuilder`, a vocabulary can be built from an arbitrary list of tweets (can be useful for testing). 
-- Although it's admittedly less useful than for other stages.

Example:
```bash
python vocab_building/vocab_builder.py twitter-datasets/preprocessed__noop/test_data.txt --meth top-k-freq -k 500
# look for the output files at vocabularies/noop__test_data/top-k-freq__500.pkl and .txt
# also creates intermediary file vocabularies/noop__test_data/full_occurrence_dict.pkl
```
