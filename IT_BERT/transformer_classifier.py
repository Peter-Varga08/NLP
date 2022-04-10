#!/usr/bin/env python

"""
Filename:  transformer_classifier.py
Description:
    This system fine-tunes an Italian BERT model to predict who (out of three subjects) is the
    post-editor of an automatically translated sentence. Predictions are cross-validated on the training set.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from simpletransformers.classification import ClassificationArgs, ClassificationModel


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_file", default='./data/IK_NLP_22_PESTYLE/train.tsv',
                        type=str, help="Location of training file.")
    parser.add_argument("-te", "--test_file", default='./data/IK_NLP_22_PESTYLE/test.tsv',
                        type=str, help="Location of test file.")
    parser.add_argument("-o", "--outfile", default='./transformer_predictions.txt',
                        type=str, help="Path to write output to.")
    parser.add_argument("-m", "--model",
                        type=str, help="Specify an already fine-tuned model")
    return parser.parse_args()


def prepare_train(train_file, include_from_scratch=False):
    """
    Extract the relevant features from the data set: MT, PE and subject.
    If include_from_scratch is False, we do not take into account the HT modality.
    """
    pairs = []
    labels = []
    with open(train_file) as train:
        train.readline()  # Skip first line: this is the header
        for line in train:
            if line.split("\t")[1][0] == "t":
                # Get a list of: [MT output, post-edited sentence, subject id]
                if include_from_scratch:
                    pairs.append([line.split("\t")[4], line.split("\t")[5]])
                    labels.append(line.split("\t")[1])
                else:
                    if line.split("\t")[4]:
                        pairs.append([line.split("\t")[4], line.split("\t")[5]])
                        labels.append(line.split("\t")[1])

    # We use only 90% of the data, in order to allow comparison with the other models
    X_train, _, y_train, _ = train_test_split(pairs, labels, test_size=0.1, random_state=42)

    return X_train, y_train


def predict_test(test_data, model, include_from_scratch=False, final=False):
    """
    Given a fine-tuned model, predict the labels (translators) from MT-PE sentence pairs.
    If include_from_scratch is False, we do not take into account the HT modality.
    """
    pairs = []
    predictions = []

    if final:  # Use the 'final' (official) test set
        with open(test_data) as test:
            test.readline()  # Skip first line: this is the header
            for line in test:
                if line.split("\t")[0][0] in "0123456789":
                    if include_from_scratch:
                        pairs.append([line.split("\t")[4], line.split("\t")[5]])
                    else:
                        if line.split("\t")[4]:
                            pairs.append([line.split("\t")[4], line.split("\t")[5]])
    else:  # Use a subset of the data for cross-validation
        pairs = test_data

    for pair in pairs:
        prediction, _ = model.predict(pair)
        predictions.append(prediction[0])

    return predictions


def get_accuracy(gold, pred):
    """
    Retrieve the number of correctly predicted post-editors
    """
    # Map labels to ints, to streamline with model predictions
    gold_ints = []
    for g in gold:
        if g == "t1":
            gold_ints.append(0)
        elif g == "t2":
            gold_ints.append(1)
        elif g == "t3":
            gold_ints.append(2)
        else:
            pass

    # Get number of correct predictions
    correct = 0
    if len(gold_ints) == len(pred):
        for g, p in zip(gold_ints, pred):
            if g == p:
                correct += 1
    else:
        print("ERROR: Gold standard and predictions are not of equal length!")

    return correct / len(gold)


def reformat_test(pairs):
    """
    Retrieve a regular list from the numpy array
    The output is a list that contains lists ([MT,PE]) with each sentence pair.
    """
    data = []
    mt = [t[0] for t in pairs]
    pe = [t[1] for t in pairs]

    for m, p in zip(mt, pe):
        data.append([m, p])

    return data


def reformat_train(pairs, labels):
    """
    Retrieve a regular list from the numpy array
    The output is a list that contains lists ([MT,PE,label]) from each sentence pair.
    """
    data = []
    mt = [t[0] for t in pairs]
    pe = [t[1] for t in pairs]
    labs = np.ndarray.tolist(labels)

    for m, p, l in zip(mt, pe, labs):
        data.append([m, p, l])

    return data


def main():

    args = create_arg_parser()
    all_predictions = []

    if args.model:  # Use an already fine-tuned model, test on the final test set
        model = ClassificationModel("bert", args.model, use_cuda=False)
        predictions = predict_test(args.test_file, model, final=True)
        all_predictions.append(predictions)
        all_predictions.append("\n")

    else:  # Fine-tune a new model, use cross-validation

        # Define model
        model_args = ClassificationArgs()
        model_args.num_train_epochs = 3
        model_args.overwrite_output_dir = True
        model_args.learning_rate = 0.000001
        model_args.do_lower_case = True
        model_args.silence = True
        model = ClassificationModel("bert", "dbmdz/bert-base-italian-uncased", args=model_args, use_cuda=False, num_labels=3)

        # Create the correct splits for cross-validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        X, y = prepare_train(args.train_file)
        X = np.array(X)
        y = np.array(y)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Prepare train set
            train_data = reformat_train(X_train, y_train)
            train_df = pd.DataFrame(train_data)
            train_df.columns = ["text_a", "text_b", "labels"]
            train_df["labels"] = train_df["labels"].replace(["t1", "t2", 't3'], [0, 1, 2])
            print(train_df)

            # Train model
            model.train_model(train_df)

            # Predict synsets
            predictions = predict_test(reformat_test(X_test), model)
            all_predictions.append(predictions)
            all_predictions.append("\n")
            print("ACCURACY: ", get_accuracy(y_test, predictions))

    # Write results to pickle file
    with open(args.outfile, 'w') as pred_file:
        for pred_list in all_predictions:
            for pred in pred_list:
                pred_file.write(str(pred))
    print("Predictions have been written to file: " + args.outfile)


if __name__ == "__main__":
    main()
