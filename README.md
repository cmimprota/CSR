# cil-spring20-project

ETHZ CIL Text Classification 2020

Meeting notes are available [here](https://demo.codimd.org/zQy-NfYSQZC0ZIy3pN_ZMA?view)

BERT git project available [here](https://github.com/huggingface/transformers)

PyTorch sentiment analysis tutorials available [here](https://github.com/bentrevett/pytorch-sentiment-analysis)

We are allowed to use pre-trained BERT. Check the link [here](https://github.com/google-research/bert)

We should perform stemming - **If somebody knows something external so that we can just import into project would be nice**)

*  More About it [here](https://www.geeksforgeeks.org/python-stemming-words-with-nltk/)
*  Check the guide [here](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)

Research papers to read can be found [here](https://docs.google.com/document/d/1-6GRa9-q5DmtTEyYLvCvwubjkC2Hquf_ndlONv_-5yI/edit?usp=sharing)

## SETUP PROJECT

- Clone the repository 
- Setup your own IDE (if not JetBrains one, please add project files to .gitignore)
- Download the full dataset [here]( http://www.da.inf.ethz.ch/files/twitter-datasets.zip) and put unzip it in your project folder 

## THE BIG PICTURE
- We start with vocabularies. We should first create full vocabularies and for that we will use a single script and input different dataset combinations.
- The only transformation that could be allowed here is data preprocessing (stemming or lemmatization). Yet **to be decided** whether to put here or in cuttings vocabularies.
- For every algorithm that we will try out we will probably need different word embeddings and we will try out different vocabularies.
- That is why for every type of cut (transformation) of full vocabulary we will have a script and we can create many cuts with different parameters. 
- These created cut vocabularies are then used and passed as argument in our training script.
- For each algorithm, we might also try different ANN structures. That is why we are separating a model in a different file.
- When in the end we run the training script, its training can take 4h or more on a leonard cluster with the full datasets. 
- We do not want to risk having a mistake in testing script that would cause our 4h training to be lost. This is why we are storing trained model as a pkl. 
- Now once we create a submission file, it can happen that we forget which model structure and vocabulary we used and that is why in next chapter I will explain how we cover this.

## GUIDE ON HOW TO IMPLEMENT ALGORITHM FROM SCRATCH
1.  Create a new folder as a subfolder of algorithms and give it a relevant name
2.  Create 3 scripts: test, train and model
3.  In the train script, the input parameters should be the training dataset and vocabulary
4.  Since for the vocabulary we performed data preprocessing, do the same for the training dataset now. (TODO: we should have a script for this)
5.  Do the word embedding to prepare dataset and transform it into input to your model
6.  Instantiate and train the model
7.  Call algorithms.helpers.save_model method and pass the trained model and the input parameters (training dataset and vocabulary)
8.  This results in creating a new subfolder results and within that another subfolder named by the timestamp
9.  In there we are saving the trained model as pkl and model structure as txt
10.  It also creates configuration_log and fills it with timestamp used for naming the subfolder in the results and corresponding training dataset and vocabulary
11.  When you create the training script, you should pass the folder name (timestamp) of the model that you want to test as parameter.
12.  Inside the script call the algorithms.helpers.load_vocabulary. This will read the configuration file and use the exact same vocabulary as for training
13.  Then you should call algorithms.helpers.load_model which again loads the correct one from the pkl file
14.  You should now input the test data to trained model in order to get the predicted labels
15.  The predicted labels should be array of -1 and 1
16.  Call call algorithms.helpers.save_submission method and pass the predicted labels to generate submission file that will be uploaded to kaggle
17.  This way we can generate many models, submission files and have a full traceability of what was used. 
18.  Once we observe the score we have obtained with different submissions we will tweak the parameters more efficiently to get better results


## PROJECT STRUCTURE

algorithms folder:
- contains subfolders for each algorithm that we will try out
- each subfolder should contain 3 files: model, test and train scripts and also configuration_log file
- each folder contains also result subfolder where we should save trained model, its structure and submission file

vocabularies folder:
- contains 3 subfolders: full, cuttings and cut
- contains a script that should be used only once for generating full vocabularies from twitter datasets and storing them in full subfolder
- contains cuttings subfolder which will do transformation and cut full vocabularies and store them in cut subfolder


## LEONHARD TUTORIAL

#### Setup environment
- Installing a Python package locally, using distutils
```shell script
command mkdir $HOME/python
command cd $HOME/python
command mkdir -p lib64/python3.7/site-packages
command echo "export PYTHONPATH=$HOME/python/lib64/python3.7/site-packages:$PYTHONPATH" >> ~/.bash_profile
command source .bash_profile
command module load gcc/6.3.0 python_gpu/3.7.4
```
- Create virtualenv
```shell script
command pip3 install --user virtualenv #install a package locally using pip
command python3 --version # check python3 version
command virtualenv --system-site-packages -p python3 ./nlp
command source ./nlp/bin/activate
command pip install --upgrade pip # do not need install --user any more
command pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
command pip list
command deactivate
```

#### Prepare for training
- Load necessary module every time login to Leonhard
```shell script
command module load module load gcc/6.3.0 python_gpu/3.7.4 # Python 3.7.4, TensorFlow 2.0.0, PyTorch 1.5.0 CUDA 10.1.243, cuDNN 7.6.4
command module load hdf5/1.10.1
```
- Activate virtualenv
```shell script
command source ./nlp/bin/activate
```

- Submit a GPU job
```shell script
command bsub -n 4 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python bert_train.py 
```

- Monitoring
```shell script
command watch -n 0.1 bpeek
```