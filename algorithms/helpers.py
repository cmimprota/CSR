import csv
import getpass
import os
import pickle
import sys

import pandas
from datetime import datetime
from torchsummary import summary

import joblib
import constants


# TODO Consider if we need any additional parameters to make it generic enough??? (example: epochs)
def save_model(trained_model, vocabulary, dataset):
    """
    Saves the 'trained_model' into the trained_model.pkl file.
    In Addition saves vocabulary and dataset (parameters) used during training in the configuration_log.csv file.
    In Addition saves the model structure (layers with inputs and outputs information) in the model_structure.txt file.
    :param trained_model: Generated with training script.
    :param vocabulary: passed as parameter of script for running the training. (selects the previously generated vocab)
    :param dataset: passed as parameter of script for running the training. (selects full vs short dataset)
    :return: void
    """
    invoking_algorithm_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    log_file_path = os.path.join(invoking_algorithm_path, "configuration_log.csv")
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w+") as config_file:
            w = csv.writer(config_file)
            w.writerow(["Author", "Timestamp", "Vocabulary", "Dataset"])

    username = getpass.getuser()
    timestamp = datetime.now().strftime("%d.%m_%H.%M.%S")

    with open(log_file_path, "a+") as config_file:
        w = csv.writer(config_file)
        w.writerow([username, timestamp, vocabulary, dataset])

    path_for_results = os.path.join(invoking_algorithm_path, "results", f"{username}-{timestamp}")
    os.makedirs(path_for_results)

    joblib.dump(trained_model, os.path.join(path_for_results, "trained_model.pkl"), compress=9)
    with open(os.path.join(path_for_results, "model_structure.txt"), "w+") as model_structure_file:
        # TODO Figure out the best structure
        # print(model)                    - (linear): Linear(in_features=714, out_features=2, bias=True)
        # summary(model, (3, VOCAB_SIZE)) - Linear-1                 [-1, 3, 2]           1,430
        # summary(model, (VOCAB_SIZE, 1)) - THIS IS A MISSMATCH
        # summary(model, (1, VOCAB_SIZE)) - Linear-1                 [-1, 1, 2]           1,430
        print(trained_model, file=model_structure_file)


def load_vocabulary(model_folder):
    """
    Reads the configuration_log.csv file to find the vocab that was used for the selected model.
    :param model_folder: passed as parameter of script for running the testing. (selects the vocab used for the model)
    :return vocab: The same one that was used for the selected model.
    """
    invoking_algorithm_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    log_file_path = os.path.join(invoking_algorithm_path, "configuration_log.csv")
    df = pandas.read_csv(log_file_path)
    username = model_folder.split("-")[0]
    timestamp = model_folder.split("-")[1]

    all_versions_of_user = df.loc[(df["Author"] == str(username)) & (df["Timestamp"] == str(timestamp))]
    vocab_filename = all_versions_of_user["Vocabulary"].values[0]
    with open(os.path.join(constants.VOCABULARIES_CUT_PATH, vocab_filename ), "rb") as inputfile:
        vocab = pickle.load(inputfile)
    return vocab


def load_model(model_folder):
    """
    Loads the 'trained_model' from the trained_model.pkl file.
    :param model_folder: passed as parameter of script for running the testing. (selects the previously generated model)
    :return 'trained_model': the one that was generated with training script.
    """
    invoking_algorithm_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    path_for_results = os.path.join(invoking_algorithm_path, "results", model_folder)
    trained_model = joblib.load(os.path.join(path_for_results, "trained_model.pkl"))
    return trained_model


def save_submission(label_predictions, model_folder):
    """
    Saves the predicted classes from 'label_predictions' into the submission.csv file.
    :param label_predictions: list of -1 or 1 obtained with by testing script.
    :param model_folder: passed as parameter of script for running the testing. (selects the previously generated model)
    :return: void
    """
    invoking_algorithm_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    path_for_results = os.path.join(invoking_algorithm_path, "results", model_folder)
    with open(os.path.join(path_for_results, "submission.csv"), "w") as f:
        f.write("Id,Prediction\n")
        for i, label in enumerate(label_predictions, start=1):
            f.write(f"{i},{label}\n")
