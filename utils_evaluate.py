#!/usr/bin/env python

"""
Filename:  evaluate.py
Description:
   This program compares system output for subject (post-editor) prediction with the gold standard (test file).

Usage example:
   python3 utils_evaluate.py -g [TEST FILE] -s [SYSTEM OUTPUT]

   where TEST FILE is the official test file sent via mail: test.tsv
   and SYSTEM OUTPUT is a .txt file with predictions, consisting of one line of predictions, for example:
   12222121111122201111222111212122111102222212122211011122112221111111111101122222
"""

import argparse

import numpy as np
import pandas as pd

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gold_standard", default='./data/IK_NLP_22_PESTYLE/test.tsv',
                        type=str, help="Official test file")
    parser.add_argument("-s", "--system_output", default='./transformer_predictions.txt',
                        type=str, help="Location of test file.")
    return parser.parse_args()


def get_gold_labels(test_file, include_from_scratch=False):
    """
    Extract the gold labels (post-editors) from the official test file
    """
    gold_labels = ""
    with open(test_file, "r") as gold_file:
        gold_file.readline()  # Skip first line: this is the header
        for line in gold_file:
            if not include_from_scratch:
                if line.split("\t")[4]:
                    if line.split("\t")[1] == "t1":
                        gold_labels += str(0)
                    elif line.split("\t")[1] == "t2":
                        gold_labels += str(1)
                    elif line.split("\t")[1] == "t3":
                        gold_labels += str(2)
                    else:
                        print("Post-editor is unknown!")
            else:
                print("We do not use translations from scratch!")
                pass  # If we include translations from scratch at some point, we can deal with it here.
    # gold_labels = pd.read_csv(test_file, header=None)
    # y = ''
    # for idx, row in gold_labels.iterrows():
    #     val = row.item()
    #     if val == 't1':
    #         y += "0"
    #     elif val == 't2':
    #         y += "1"
    #     elif val == "t3":
    #         y += "2"
    # return y

    return gold_labels


def get_correct_predictions(gold, pred):
    """
    Retrieve the number of correctly predicted post-editors
    """
    correct = 0
    if len(gold) == len(pred):
        for g, p in zip(gold, pred):
            if g == p:
                correct += 1
    else:
        print("ERROR: Gold standard and predictions are not of equal length!")

    return correct


def main():
    args = create_arg_parser()

    # Load system predictions and gold standard labels
    gold = get_gold_labels(args.gold_standard)
    with open(args.system_output, "r") as prediction_file:
        pred = prediction_file.readline()

    pred = pred.rstrip()
    # Get the number of correct predictions
    correct_predictions = get_correct_predictions(gold, pred)

    # Print error report
    print('Number of predictions:', len(pred))
    print("ABSOLUTE NUMBER OF CORRECT PREDICTIONS: ", correct_predictions)
    print("ACCURACY: ", correct_predictions / len(gold))


if __name__ == "__main__":
    main()
