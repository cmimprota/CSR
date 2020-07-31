# Model Architecture

## CharCNN
CharCNN is mainly based on the model of Zhang's [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).

## Character-level Sequence Representation with Resdiual Block (CSR-Res)
Inspired by CharCNN, we modify the original CharCNN and add residual layers. The bottleneck block is introduced which consists of
convolutional and maxpooling layers. The residual block only contains one convolutional layer. For the residual block which increases the channels,
a two stride convolutional layer is applied. In order to match the dimensions, we also apply a convolutional layer with 
kernel size and stride equal to one and two respectively to the identity shortcut which we call downsampling layer. Batch
normalization (BN) is implemented right after each convolution and before Relu activation. We use character-level sequence 
representation with residual block (CSR-Res) as the character embedding model, performing the same as word embedding model like BERT.


# Classifier
Fully connected layers and multiple-layer RNN is applied as the classifier.

# Structure
* `src/` - source data files required for training/testing/validating.
    * `models/` - model implementations,
    * `utils/` - utility methods.
* `alphabet.json` - list of characters for one hot encoding.
* `main.py` - training process file.  

# Dataset Format
Each sample looks like:
        
        "Tweet", "Label"
        
Samples are separated by newline.

Example (0 for negative and 1 for positive):
    
    "vinco tresorpack 6 ( difficulty 10 of 10 object : disassemble and reassemble the wooden pieces this beautiful wo ... <url>", "0"
    "<user> i dunno justin read my mention or not . only justin and god knows about that , but i hope you will follow me #believe 15", "1"

# Train
```shell script
python main.py -h
```
You will see
```shell script
usage: main.py [-h] [--train_path DIR] [--val_path DIR] [--lr LR]
               [--epochs EPOCHS] [--batch_size BATCH_SIZE]
               [--max_norm MAX_NORM] [--optimizer OPTIMIZER] [--class_weight]
               [--dynamic_lr] [--milestones MILESTONES [MILESTONES ...]]
               [--decay_factor DECAY_FACTOR] [--alphabet_path ALPHABET_PATH]
               [--l0 L0] [--shuffle] [--dropout DROPOUT]
               [-kernel_num KERNEL_NUM] [-kernel_sizes KERNEL_SIZES]
               [--num_workers NUM_WORKERS] [--cuda] [--verbose]
               [--continue_from CONTINUE_FROM] [--checkpoint]
               [--checkpoint_per_batch CHECKPOINT_PER_BATCH]
               [--save_folder SAVE_FOLDER] [--log_config] [--log_result]
               [--log_interval LOG_INTERVAL] [--val_interval VAL_INTERVAL]
               [--save_interval SAVE_INTERVAL]

CIL text sentiment classifier training

optional arguments:
  -h, --help            show this help message and exit
  --train_path DIR      path to training data csv [default:
                        data/ag_news_csv/train.csv]
  --val_path DIR        path to validation data csv [default:
                        data/ag_news_csv/test.csv]

Learning options:
  --lr LR               initial learning rate [default: 0.0001]
  --epochs EPOCHS       number of epochs for train [default: 200]
  --batch_size BATCH_SIZE
                        batch size for training [default: 64]
  --max_norm MAX_NORM   Norm cutoff to prevent explosion of gradients
  --optimizer OPTIMIZER
                        Type of optimizer. SGD|Adam|ASGD are supported
                        [default: Adam]
  --class_weight        Weights should be a 1D Tensor assigning weight to each
                        of the classes.
  --dynamic_lr          Use dynamic learning schedule.
  --milestones MILESTONES [MILESTONES ...]
                        List of epoch indices. Must be increasing.
                        Default:[5,10,15]
  --decay_factor DECAY_FACTOR
                        Decay factor for reducing learning rate [default: 0.5]

Model options:
  --alphabet_path ALPHABET_PATH
                        Contains all characters for prediction
  --l0 L0               maximum length of input sequence to CNNs [default:
                        1014]
  --shuffle             shuffle the data every epoch
  --dropout DROPOUT     the probability for dropout [default: 0.5]
  -kernel_num KERNEL_NUM
                        number of each kind of kernel
  -kernel_sizes KERNEL_SIZES
                        comma-separated kernel size to use for convolution

Device options:
  --num_workers NUM_WORKERS
                        Number of workers used in data-loading
  --cuda                enable the gpu

Experiment options:
  --verbose             Turn on progress tracking per iteration for debugging
  --continue_from CONTINUE_FROM
                        Continue from checkpoint model, we can use
                        /content/drive/My
                        Drive/cil/models_CharResCNN/CharResCNN_best.pth.tar
  --checkpoint          Enables checkpoint saving of model
  --checkpoint_per_batch CHECKPOINT_PER_BATCH
                        Save checkpoint per batch. 0 means never save
                        [default: 10000]
  --save_folder SAVE_FOLDER
                        Location to save epoch models, training configurations
                        and results.
  --log_config          Store experiment configuration
  --log_result          Store experiment result
  --log_interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 1]
  --val_interval VAL_INTERVAL
                        how many steps to wait before vaidation [default: 200]
  --save_interval SAVE_INTERVAL
                        how many epochs to wait before saving [default:1]
```

