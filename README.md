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
3.  Input your nethz password when prompted (WILL BE ASKED IF SSH KEYS ARE NOT SETUP)
4.  Type Yes to agree to the terms (WILL BE ASKED ONLY ONCE IN LIFETIME)

Congratulations, you are there!

#### Setup environment (**Do this only once, before training for the first time**)
- Setup SSH keys
You do not need to know what SSH keys are, you need to know what it allows. 
It allows you to do git clone and git push without typing credentials every time.
Lets do the following set of commands on Leonhard first!

```
ssh-keygen
```

It will ask to save key at /cluster/home/username/.ssh/id_rsa - press enter.
It will ask for passphrase - leave it empty and press enter twice.
Great, SSH is created and now we can attach it to your gitlab account.

```
cat ~/.ssh/id_rsa.pub 
```
   
1.  You will see a long string output in your terminal. Please select it and right click copy
2.  Go to [https://gitlab.ethz.ch/](https://gitlab.ethz.ch/) and log in
3.  Click in top right corner and select settings
4.  From the left side menu click on SSH keys
5.  Paste the long text in the big textbox under Key
6.  In Title textbox enter Leonhard and press Add Key button below

    Great, now go and repeat the same procedure on your local computer (if you do not already have the ssh and did not put it in the gitlab).
If you execute the following command on your local coputer it will allow to login to Leonhard or copy files using scp command without prompting for credentials:

```
cat $HOME/.ssh/id_rsa.pub | ssh username@login.leonhard.ethz.ch "cat - >> .ssh/authorized_keys"
```

Finally this commands should be executed back on the Leonhard cluster to make everything work:
```
chmod 700 $HOME/.ssh
chmod 600 ~/.ssh/authorized_keys
```

- Installing a Python package locally, using distutils (FULL GUIDE [HERE](https://scicomp.ethz.ch/wiki/Python#Installing_a_Python_package.2C_using_PIP))

We are using a cluster, meaning that there is python already installed that we are supposed to simply load as module.
However, python interpreter is looking for the modules (libraries) under the root location that we do not have access to.
We cannot sudo and simply install libraries that we want. This is why later we will install virtual environment (venv)
In order to be able to reference our own scripts using import, we will add PYTHONPATH the root of the project location.

```
echo "export PYTHONPATH=$HOME/cil-spring20-project" >> ~/.bash_profile
source ~/.bash_profile
module load gcc/6.3.0 python_gpu/3.7.4
```

- Setup the project

In order for SSH keys to work, clone the repository using this specific URL and not the https one.

```
cd ~
git clone git@gitlab.ethz.ch:mstevan/cil-spring20-project.git
```

If the authenticity of host 'gitlab.ethz.ch (129.132.202.219)' can't be established is prompted, type yes and click enter

- Create virtualenv

Typically all the modules (libraries) that we need should be in requirements.txt and we can use it to quickly download them in venv 

```shell script
cd ~/cil-spring20-project/
pip3 install --user virtualenv
virtualenv -p python3 venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
deactivate
```


#### Prepare for training (**Do this every time you want to train**)
- Outside of Leonhard copy the necessary files that are missing (if not already coppied before) - example .gitignored from project such as datasets 

```
scp -r path-to-the-directory-on-local-pc/cil-spring20-project/twitter-datasets nethzusername@login.leonhard.ethz.ch:~/cil-spring20-project/
```

- Login to Leonhard as instructed before and run everything on it from now on

- Get latest version of your code

```
cd ./cil-spring20-project/ && git pull && cd ~
```

- Load necessary module 

```
module load python_gpu/3.7.4
```

- Activate virtualenv
```
source ./cil-spring20-project/venv/bin/activate
```


#### Train (**You can submit as many training jobs you want within single session**)
- Submit a GPU job (for S this managed to timeout)
```
cd ~/cil-spring20-project/algorithms/baseline_one/ && bsub -n 4 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python bow_train.py -d train-short -v cut-vocab-test-frequency-20.pkl && cd ~
``` 

- Submit a job without GPU (for S this did not timeout for the same training model)
```
bsub -n 20 -R "rusage[mem=4500]" python ~/cil-spring20-project/algorithms/baseline_one/bow_train.py -d train-full -v cut-vocab-test-frequency-20.pkl
```

- Monitoring
```
watch -n 0.1 bpeek
```

- Checking the log

    Job's output is written into a file named lsf.oJobID in the directory where you executed bsub. 
    If you used the command above, it will end up under your algorithm folder.
    If you want to change this, you can either use the -o option. 
    Or you can simply use the Submit a GPU job command from within some other folder (command is precreated to support this). 
    More information is available [here](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#Output_file)

#### Push results to git (**You can select which ones you want to push yourself**)
- Using helper functions creates a results subfolder under your algorithm folder
- Each training attempt creates folder with unique name and stores trained model there while test creates submission file there
- Decide which ones you want to add using this command for each file **git add filename** (be careful not to include unnecessary files)
- git commit -m "name-of-algorithm-new-results"
- git push


## Troubleshooting
#### Accessing Leonhard Cluster after cyber attack
Full guide available [here](https://scicomp.ethz.ch/wiki/Reopening_of_Euler_and_Leonhard_(May_2020))
1. Change AAI (LDAP) password. Changing VPN and Active Directory does not have any effect
2. Create a strong new ssh key, connect to VPN and copy the public key to Leonhard 
```
ssh-keygen -t ed25519
ssh-copy-id -i $HOME/.ssh/id_ed25519.pub username@login.leonhard.ethz.ch
```
In case you have problems with git Authentication after password change, follow these steps for Windows:
1. Click Start
2. Type: Credential Manager (On Windows 10, this is under "Start->Settings". Then search for "Credential Manager")
3. See the Windows Credentials Manager shortcut and double-click it to open the application.
4. Once the app is open, click on the Windows Credentials tab.
5. Locate the credentials that you want to remove/update, they will start with "git:" and might begin with "ada:"
6. Click on the credential entry, it will open a details view of the entry.
7. Click Edit and type your new 

In case you have problems with PyCharm after password change, follow these steps:
1. Locate and delete .IntelliJIdea14\config\options\security.xml
2. Open PyCharm and select File -> Invalidate Caches/Restart from the main menu.


#### ModuleNotFoundError: No module named 'algorithms'
This can happen if you are not using a smart IDE that figures out things by itself. 
The problem is that if you are trying to run a python file that is stored in a subfolder of subfolder... it is unable to load these modules.
Two possible fixes:
1.  In the scrypt that you are running insert following code at the beginning. 

```
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

Not that number of os.path.dirnames have to match the distance of the subfolder from the root folder
2.  Modify your PYTHONPATH variable. This variable tells the python interpreter where to look for the modules. 
Variable can contain multiple folders that are separated by semicolon (:). This modification can sometimes be very painful but it short can be done with following command:

```
echo "export PYTHONPATH=$HOME/cil-spring20-project:$PYTHONPATH" >> ~/.bash_profile
```

The command means take the current value of PYTHONPATH variable, append the $HOME/cil-spring20-project and append this whole command with export to ~/.bash_profile.
It often happens that people expect it to become available immediately. But it does not work like that. In order for your variable to be effective you need to use:

```
source ~/.bash_profile
```

Because it doesn't work with the command above, it often happens that you call the echo export couple of times. However that creates a mess which can be verified with:

```
echo $PYTHONPATH
```

If this is a case, do not worry, everything can be easily fixed. You need to manually modify the ~/.bash_profile. 
It is perfectly fine that you open it with **nano ~/.bash_profile** (if on cluster remember to **module load nano** before that).
All you have to do is to delete all rows starting with **export PYTHONPATH** and leave a single one. 
If your single one contains multiple same folders separated with semicolon you can delete them as well.
Do not forget to **source ~/.bash_profile**

#### Results folder created in a weird places

Even though most of bugs have been fixed, it can happen that results folder with your trained model ends up in a weird location due to the way helper.py script was created.
In this case please locate the results folder and copy it to the repository at the correct location

```
find . -name results
cp -a previous_output/results/. ~/cil-spring20-project/algorithms/your_algorithm_folder/results/
rm -rf previous_output/results
```