import argparse
import sys
import os

def mkdirs(dir_path, verbose=True):
  try:
    os.makedirs(dir_path)
  except OSError:
    if verbose:
      print ("Creation of the directory %s failed" % dir_path)

def parse():
  parser = argparse.ArgumentParser()  
  parser.add_argument("-r", "--result_path", help="Directory where to save results.")
  parser.add_argument("-o", "--oversampling", default=False, help="Whether to oversample training data.")
  parser.add_argument("-k", "--kfold", default=False, help="Whether to do K(=10)-fold cross-validation.")
  parser.add_argument("-f", "--features", help="Which features to include:\n\
    - log\n\
    - ling\n\
    - both."
  )
  try:
    args = parser.parse_args()
  except Exception as e:
    print(e)
    parser.print_help()
    sys.exit(0)
  result_path = args.result_path
  mkdirs(result_path)
  features = args.features.lower()
  oversampling = args.oversampling
  kfold = args.kfold
  return result_path, features, oversampling, kfold