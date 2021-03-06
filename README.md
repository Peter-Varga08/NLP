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

* __Statistical model__:
  * Training a RandomForest model and obtaining test predictions on logging features:
    * `(cd Statistical_and_ML_models/ && python3 run.py -e numeric -m rf -s -t)`
  * Training a RandomForest model and obtaining test predictions on linguistic features:
    * `(cd Statistical_and_ML_models/ && python3 run.py -e linguistic -m rf -s -t)`
  * Training a RandomForest model and obtaining test predictions on both features:
    * `(cd Statistical_and_ML_models/ && python3 run.py -e combined -m rf -s -t)`
    
  * NOTE: Training the LinearRegression model can be done using the same commands, with `-m` being supplied with a value of `lr`:


* __Neural network__: see the readme file in the NeuralNetwork folder

* __IT BERT__:
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
