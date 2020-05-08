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
#### FULL TUTORIAL AVAILABLE [HERE](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters)
#### Login into Leonhard environment (**Every time you want to do something on Leonhard**)
Every student should have his/her personal home directory on leonhard by default.
This means that you can do whatever you want, independet of others!
1.  VPN to ETH network
2.  ssh nethzusername@login.leonhard.ethz.ch.
3.  Input your nethz password when prompted
4.  Type Yes to agree to the terms

Congratulations, you are there!

#### Setup environment (**Do this only once, before training for the first time**)
- Installing a Python package locally, using distutils
```shell script
mkdir $HOME/python
cd $HOME/python
mkdir -p lib64/python3.7/site-packages
echo "export PYTHONPATH=$HOME/python/lib64/python3.7/site-packages:$PYTHONPATH" >> ~/.bash_profile
source ~/.bash_profile
module load gcc/6.3.0 python_gpu/3.7.4
```

- Setup the project
```shell script
cd ~
git clone https://gitlab.ethz.ch/mstevan/cil-spring20-project.git
```
    Input your nethz username and password when prompted

- Create virtualenv
```shell script
cd ~/cil-spring20-project/
pip3 install --user virtualenv
virtualenv --system-site-packages -p python3 venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
deactivate
```


#### Prepare for training (**Do this every time you want to train**)
- Outside of Leonhard copy the necessary files that are missing (if not already coppied before) - example .gitignored from project such as datasets 
```shell script
scp -r path-to-the-directory-on-local-pc/cil-spring20-project/twitter-datasets nethzusername@login.leonhard.ethz.ch:~/cil-spring20-project/
```
    Input your nethz password when prompted
    
- Login to Leonhard as instructed before and run everything on it from now on
- Get latest version of your code
```shell script
cd ./cil-spring20-project/ && git pull && cd ~
```

- Load necessary module 
```shell script
module load gcc/6.3.0 python_gpu/3.7.4
module load hdf5/1.10.1
```

- Activate virtualenv
```shell script
source ./cil-spring20-project/venv/bin/activate
```


#### Train (**You can submit as many training jobs you want within single session**)
- Submit a GPU job
```shell script
(cd ~/cil-spring20-project/ && bsub -n 4 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ~/cil-spring20-project/algorithms/baseline_one/bow_train.py -d train-short -v cut-vocab-test-frequency-20.pkl)
``` 

- Monitoring
```shell script
watch -n 0.1 bpeek
```

- Checking the log

    Job's output is written into a file named lsf.oJobID in the directory where you executed bsub.
If you want to change this, you can either use the -o option. 
Or you can simply use the Submit a GPU job command from within some other folder (command is precreated to support this). 
More information is available [here](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#Output_file)

#### Push results to git (**You can select which ones you want to push yourself**)
- Using helper functions creates a results subfolder under your algorithm folder
- Each training attempt creates folder with unique name and stores trained model there while test creates submission file there
- Decide which ones you want to git add (be careful not to include unnecessary files)
- git commit -m "name-of-algorithm-new-results"
- git push