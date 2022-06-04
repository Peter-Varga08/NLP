# Comparing Statistical and Machine Learning Approaches to Identify Post-Editors from Logging and Linguistics Features: an Investigative Stylometry Study

## Description
This repository contains the code used for the report *Comparing Statistical and Machine Learning Approaches to Identify Post-Editors from Logging and Linguistics Features: an Investigative Stylometry Study*. This report was written for the final project of the course *Natural Language Processing* at the University of Groningen.

The models are contained in the folders *IT_BERT, NeuralNetwork* and *Statistical_and_ML_models*. The linguistic features that we extracted are in *Linguistic_features*, accompanied by a legend with a more detailed description of the features. 

#### Data

Since the data set is currently not yet publicly available, we do not provide it in this repository. Its full description can be found on its [Huggingface Dataset card](https://huggingface.co/datasets/GroNLP/ik-nlp-22_pestyle)


## Installation

1. Install dependencies with `pip install -r requirements.txt`
2. If you wish to use our fine-tuned Italian BERT model, you can download it from [here](https://drive.google.com/drive/folders/1pAsYmxCd2ch0zrofpH-C-spYXEsJP7M7?usp=sharing).

## Usage

* ##__Statistical model__:
  * Training a RandomForest model and obtaining test predictions on logging features:
    * `(cd Statistical_and_ML_models/ && python3 run.py -e numeric -m rf -s -t)`
  * Training a RandomForest model and obtaining test predictions on linguistic features:
    * `(cd Statistical_and_ML_models/ && python3 run.py -e linguistic -m rf -s -t)`
  * Training a RandomForest model and obtaining test predictions on both features:
    * `(cd Statistical_and_ML_models/ && python3 run.py -e combined -m rf -s -t)`
    
  * NOTE: Training the LinearRegression model can be done using the same commands, with `-m` being supplied with a value of `lr`:


* ##__Neural network__:
  The neural network model implemented in this work consists of two hidden layers of size 512 and 256 units, respectively. The output layer consists of three neurons with a softmax activation. Batch normalization, dropout, gradient clipping and early stopping are applied as regularizations.

  This folder contains 3 subfolders, each containing model weights, features importance plots (retrieved with Integrated Gradients), and the predictions on the test set:
  - log: just logging features setting
    - ling: just linguistic features setting
    - both: combined features setting

  Following, the instruction to run the scripts.
  
  ### Important: before running, upload the dataset folder here (IK_NLP_22_PESTYLE/)
  
  To train the model:
  
  `python3 training.py -r both -k 1 -f both`
  
  To perform Integrated Gradients:
  
  `python3 training.py -r ling -f ling`

  To evaluate the model on the test set:

  `python3 get_nn_predictions.py -r log -f log`
  
  `python3 evaluating.py -g gold_file_path -s pred_file_path`

  The arguments are:
  - r: path to folder where to save the model or upload it from
  - k: whether to run a 10-fold cross-validation
  - f: features types (i.e., log, ling, both)
  - g: path to the file containing the ground truth
  - s: path to the file containing the system outputs (predictions)


* ##__IT BERT__:
  * Fine-tuning a model:
  
    `python3 transformer_classifier.py -tr IK_NLP_22_PESTYLE/train.tsv -te IK_NLP_22_PESTYLE/test.tsv -o results_file.txt`
  * Making predictions with an already fine-tuned model (such as [this one](https://drive.google.com/drive/folders/1pAsYmxCd2ch0zrofpH-C-spYXEsJP7M7?usp=sharing)):
  
    `python3 transformer_classifier.py -tr IK_NLP_22_PESTYLE/train.tsv -te IK_NLP_22_PESTYLE/test.tsv -o results_file.txt -m outputs/checkpoint-237-epoch-3`




## Evaluation

For evaluating the output of a model on the official test set, we use *utils_evaluate.py*. It takes as input a .txt-file with predictions (0, 1 and 2 for subjects t1, t2 and t3) and returns the accuracy. The following command can be run:

* Example of obtaining test accuracy of the classifier (i.e. RandomForest) model having used only the linguistic features
  - `python3 utils_evaluate.py -g IK_NLP_22_PESTYLE/IK_NLP_22_PESTYLE/test.tsv -s ./Statistical_and_ML_models/predictions/classifier_linguistic_predictions.txt`
* Example of obtaining test accuracy of the regressor (i.e. LinearRegression) model having used only the linguistic features
  - `python3 utils_evaluate.py -g IK_NLP_22_PESTYLE/IK_NLP_22_PESTYLE/test.tsv -s ./Statistical_and_ML_models/predictions/regressor_linguistic_predictions.txt`
