## Usage

Only use `data_preprocessor.py`; all the others are auxiliary files.

The script `data_preprocessor.py` allows to preprocess a raw tweets dataset, into file `twitter-datasets/preprocessed__<preproc_method>/<base_data>.txt`.

The code follows the same structure as other stages of the pipeline, e.g `tweet_embedder`. As a free benefit of the code structure, through the exported class `DataPreprocessor`, an arbitrary tweet can be preprocessed (can be useful for testing). 
-- Although it's admittedly much less useful than for other stages.

Example:
```bash
python data_preprocessing/data_preprocessor.py twitter-datasets/train_pos_dummy.txt --method xcb
# look for the output file at data_preprocessing/results/preprocessed__xcb/train_pos_dummy.txt
```
