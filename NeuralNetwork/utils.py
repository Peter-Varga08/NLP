import argparse
import sys
import os


def mkdirs(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        print("Creation of the directory %s failed" % dir_path)


def get_files(texts, filename):
    for i, text in enumerate(texts):
        with open(filename % i, 'w') as txtfile:
            txtfile.write(text)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path", help="Directory where to save results.")
    parser.add_argument("-o", "--oversampling", default=False, help="Whether to oversample training data.")
    parser.add_argument("-k", "--kfold", default=False, help="Whether to do K(=10)-fold cross-validation.")
    parser.add_argument("-f", "--features", help="Which features to include", choices=['log', 'ling', 'both'])

    try:
        args = parser.parse_args()
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(0)

    mkdirs(args.result_path)
    features = args.features.lower()

    return args.result_path, features, args.oversampling, args.kfold
