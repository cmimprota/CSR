# cil-spring20-project

ETHZ CIL Text Classification 2020

We are allowed to use pre-trained BERT. Check the link [here](https://github.com/google-research/bert)

We should perform stemming - **If somebody knows something external so that we can just import into project would be nice**)

*  More About it [here](https://www.geeksforgeeks.org/python-stemming-words-with-nltk/)
*  Check the guide [here](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)

Research papers to read can be found [here](https://docs.google.com/document/d/1-6GRa9-q5DmtTEyYLvCvwubjkC2Hquf_ndlONv_-5yI/edit?usp=sharing)

## SETUP PROJECT

- Clone the repository 
- Setup your own IDE (if not JetBrains one, please add project files to .gitignore)
- Download the full dataset [here]( http://www.da.inf.ethz.ch/files/twitter-datasets.zip) and put unzip it in your project folder 

## PROJECT STRUCTURE

algorithms folder:
- contains subfolders for each algorithm that we will try out
- each subfolder should contain a file that will be used to train and generate model
- we should save this stage this model as its training can take 4h on a cluster
- after that we can use the staged modes for testing accuracy and submitting the score

vocabularies folder:
- contains a script that should be used only once for generating full vocabularies from twitter datasets and storing them in full subfolder
- contains cuttings subfolder which will do transformation and cut full vocabularies and store them in cut folder

kaggle-helpers:
- only downloaded from kaggle. Yet to be discovered what its usage is and how to apply it