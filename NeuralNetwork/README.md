# Neural Network
The neural network model implemented in this work consists of two hidden layers of size 512 and 256 units, respectively. The output layer consists of three neurons with a softmax activation. Batch normalization, dropout, gradient clipping and early stopping are applied as regularizations.

This folder contains 3 subfolders, each containing model weights, features importance plots (retrieved with Integrated Gradients), and the predictions on the test set:
- log: just logging features setting
- ling: just linguistic features setting
- both: combined features setting

Following, the instruction to run the scripts.

## Important: before running, upload the dataset folder here (IK_NLP_22_PESTYLE/)

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
