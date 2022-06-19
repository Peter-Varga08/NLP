"""
Example usage: python train.py ML_CLF_RF
"""

import argparse
from typing import Union
from numpy.typing import NDArray

import numpy as np
import pandas as pd
import wandb
from scikeras.wrappers import KerasClassifier
from simpletransformers.config.model_args import ClassificationArgs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer

import utils_pipeline
from enums import Split
from model import ClassificationTransformer, NeuralNetwork, RegressionModel

MLModel = Union[RandomForestClassifier, LinearSVC, RegressionModel, KerasClassifier]

def train_kfold(model: MLModel, X: NDArray, y: NDArray, encoder: LabelBinarizer):
    """Train a model using logging/linguistic features with k-fold split."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        y_train_encoded, y_valid_encoded = encoder.transform(y[train_index]), encoder.transform(y[valid_index])
        X_train_scaled, X_valid_scaled = utils_pipeline.scaling(X_train, X_valid)

        # TODO: make sure all models share same interface
        model.fit(X_train_scaled, y_train_encoded)
        y_preds = model.predict(X_valid_scaled)
        y_preds_argmax = np.array([np.argmax(y) for y in y_preds])
        y_valid_argmax = np.array([np.argmax(y) for y in y_valid])

        results = {'classification_report': classification_report(y_valid_encoded, y_preds),
                   'accuracy_score': accuracy_score(y_preds, y_valid_encoded)}
        wandb.log(results)
        wandb.sklearn.plot_precision_recall(y_valid_argmax, y_preds_argmax, ["editor"])
        wandb.sklearn.plot_feature_importances(model, df_train.columns)
    # TODO: save models

def gridsearch_cv(model: MLModel, X: NDArray, y: NDArray):
    """Train a model using GridSearchCV."""
    X_train, y_train, X_valid, y_valid = train_test_split(X, y, train_size=0.9, shuffle=True, stratify=True)
    X_train_scaled, X_valid_scaled = utils_pipeline.scaling(X_train, X_valid)
    y_train_encoded, y_valid_encoded = y_encoder.transform(y_train), y_encoder.transform(y_valid)
    model.fit(X_train_scaled, y_train_encoded)
    print(sorted(model.cv_results_.keys()))
    pass

def train_bert(model: ClassificationTransformer, data: pd.DataFrame):
    pass

def train_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # |-----------------------------------------|
    # |            BASE ARGUMENTS               |
    # |-----------------------------------------|
    parser.add_argument("-r", "--result_path", default='./transformer_predictions.txt',
                        help="Directory where to save results.")
    parser.add_argument("-tr", "--train_file", default='../data/',
                        type=str, help="Location of training file.")
    parser.add_argument("-f", "--features", help="Which features to include", choices=["log", "ling", "both"])
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to enable or disable logging.')
    parser.add_argument('--grid', action='store_true')
    # |-----------------------------------------|
    # |            BERT MODEL                   |
    # |-----------------------------------------|
    bert_parser = subparsers.add_parser("BERT")
    bert_parser.add_argument('--subparser_bert', default=True)
    bert_parser.add_argument("-c", "--bert_config", type=str, default='dbmdz/bert-base-italian-uncased',
                             help="Specify a model configuration from HuggingFace.")
    # |-----------------------------------------|
    # |            NEURAL NETWORK               |
    # |-----------------------------------------|
    # Specified parameters based training
    nn_parser = subparsers.add_parser("NN")
    nn_parser.add_argument('-es', '--early_stopping', type=int, help="Patience of early stopping.", default=100)
    nn_parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    nn_parser.add_argument('-b', '--batch_size', type=int, default=16)
    nn_parser.add_argument('-ep', '--epochs', type=int, default=50)
    nn_parser.add_argument('-d', '--dropout', type=float, default=0.1)
    nn_parser.add_argument('-c', '--clipvalue', type=float, default=0.5)
    nn_parser.add_argument('-o', '--optimizer', type=str, default="Adam")
    # |-----------------------------------------|
    # |            REGRESSION ML MODELS         |
    # |-----------------------------------------|
    # TODO: add separate subparser for each regressor; writing a wrapper just to save
    # 10 lines of code is not worth it
    regressor_parser = subparsers.add_parser("ML_REG")
    # Common regressor params
    regressor_parser.add_argument('-m', '--model', type=str, default=None, required=True, choices=['lr', 'rr', 'en'],
                                  help="'lr': LinearRegression, 'rr': Ridge, 'en': ElasticNet")
    regressor_parser.add_argument('-n', '--normalize', action='store_true', help="Subtract mean and divide by L2-norm.")
    # LinearRegression() params
    regressor_parser.add_argument('--n_jobs', type=int, default=-1)
    # Ridge() and ElasticNet() params
    regressor_parser.add_argument('-a', '--alpha', type=Union[float, int], default=1,
                                  help="Control L2 term by multiplication with alpha, in [0, inf].")
    # ElasticNet() params
    regressor_parser.add_argument('-l1', '--l1_ratio', type=float, default=0.5,
                                  help='For l1=0 the penalty is an L2 penalty. For l1=1 it is an L1 penalty.')
    regressor_parser.add_argument('--max_iter', type=int, default=1000)
    regressor_parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for optimization.')
    regressor_parser.add_argument('--selection', type=str, default='random',
                                  help='Setting ‘random’ often leads to significantly faster convergence,'
                                       ' especially when tol is higher than 1e-4.')
    # |-----------------------------------------|
    # |           CLASSIFIER ML MODELS          |
    # |-----------------------------------------|
    # 1) LinearSVC
    svc_parser = subparsers.add_parser("ML_CLF_SVC")
    svc_parser.add_argument('--penalty', type=str, choices=["l1", "l2"])
    svc_parser.add_argument('--loss', type=str, choices=["hinge", "squared_hinge"])
    svc_parser.add_argument('--C', type=float, help="Inversely proportional regularization parameter.", default=1.0)
    svc_parser.add_argument('--max_iter', type=int, default=1000)
    # 2) RandomForest
    randomforest_parser = subparsers.add_parser("ML_CLF_RF")
    randomforest_parser.add_argument('-subparser', '--subparser_ml_clf_rf', default=True)
    randomforest_parser.add_argument('-ne', '--n_estimators', type=int, default=100,
                                     help='Number of estimators in the forest.', )
    randomforest_parser.add_argument('--max_depth', type=int, default=10,
                                     help="None means expansion until all leaves are pure,"
                                          " or contain less than min_samples")
    randomforest_parser.add_argument('--min_samples', type=int, default=2)
    return parser.parse_args()




if __name__ == "__main__":
    wandb.init(project="NLP", entity="petervarga")
    args = train_parser()
    wandb.config.update(args)

    # TODO: NeuralNetwork
    param_grid = {'NeuralNetwork': {},
                  'Regression': {'alpha': [0.5, 1, 2, 5, 10],
                                 'l1_ratio': [0.1, 0.25, 0.5, 0.75, 1],
                                 'max_iter': [100, 500, 1000, 2000],
                                 'tol': [1e-5, 5e-5,
                                         1e-4, 5e-4,
                                         1e-3, 5e-3,
                                         1e-2, 5e-2],
                                 'selection': ['cyclic', 'random']},
                  'RandomForest': {'n_estimators': [50, 100, 200, 500],
                                   'max_depth': [3, 5, 10, 15],
                                   'min_samples_split': [2, 3, 4, 5, 6, 7]},
                  'LinearSVC': {'C': [1, 1.5, 5, 10],
                                'tol': [1e-5, 5e-5,
                                        1e-4, 5e-4,
                                        1e-3, 5e-3,
                                        1e-2, 5e-2],
                                'loss': ['hinge', 'squared_hinge'],
                                'penalty': ['l1', 'l2'],
                                'max_iter': [100, 250, 500, 1000, 2000]}}

    # |--------------------------|
    # |       MODEL LOADING      |
    # |--------------------------|
    # BERT
    # ---------------------------------------------------------------------
    if hasattr(args, 'subparser_bert'):
        # Define model
        model_args = ClassificationArgs()
        model_args.num_train_epochs = 3
        model_args.overwrite_output_dir = True
        model_args.learning_rate = 0.000001
        model_args.do_lower_case = True
        model_args.silence = True
        model = ClassificationTransformer(args.bert_config, model_args)

        # TODO: Add BERT training steps
        # train_bert()
    else:
        # Neural Network
        # ---------------------------------------------------------------------
        if hasattr(args, 'subparser_nn'):
            if not args.grid:
                nn = NeuralNetwork(input_len=args.input_len, lr=args.lr, clipvalue=args.clip_value)
                # TODO: fix
                model = KerasClassifier(nn.model,
                                        batch_size=args.batch_size, epochs=args.epochs, verbose=1)
            else:
                # TODO: add GridSearchCV to KerasCLassifier
                model = None
        # Regressor
        # ---------------------------------------------------------------------
        elif hasattr(args, 'subparser_ml_reg'):
            if not args.grid:
                model = RegressionModel(args.model,
                                        normalize=args.normalize,
                                        n_jobs=args.n_jobs,
                                        alpha=args.alpha,
                                        l1_ratio=args.l1_ratio,
                                        precompute=True,
                                        max_iter=args.max_iter,
                                        tol=args.tol,
                                        selection=args.selection)
            else:
                model = GridSearchCV(estimator=RegressionModel(args.model),
                                     param_grid=param_grid["Regression"],
                                     cv=10,
                                     n_jobs=args.n_jobs,
                                     scoring=[f1_score, accuracy_score])
        # SVC classifier
        # ---------------------------------------------------------------------
        elif hasattr(args, 'subparser_ml_clf_svc'):
            if not args.grid:
                model = LinearSVC(C=args.C,
                                  tol=args.tol,
                                  loss=args.tol,
                                  penalty=args.penalty,
                                  max_iter=args.max_iter)
            else:
                model = GridSearchCV(estimator=LinearSVC(),
                                     param_grid=param_grid["LinearSVC"],
                                     cv=10,
                                     n_jobs=-args.n_jobs,
                                     scoring=['f1_samples', 'accuracy'])
        # RandomForest classifier
        # ---------------------------------------------------------------------
        elif hasattr(args, 'subparser_ml_clf_rf'):
            if not args.grid:
                model = RandomForestClassifier(n_estimators=args.n_estimators,
                                               max_depth=args.max_depth,
                                               min_samples_split=args.min_samples,
                                               random_state=42,
                                               n_jobs=-1)
            else:
                model = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                                     param_grid=param_grid['RandomForest'],
                                     cv=10,
                                     scoring=['accuracy'],
                                     verbose=1)
        else:
            raise SyntaxError("In order to perform training, specification of subparser is required.")

        # |--------------------------|
        # |       DATA LOADING       |
        # |--------------------------|
        if args.features == 'ling':
            print("Loading linguistic features only...")
            df_train = utils_pipeline.get_ling_feats(Split.TRAIN)
        elif args.features == 'both':
            print("Loading linguistic and logging features...")
            df_train_log = utils_pipeline.load_dataframe(split=Split.TRAIN, only_numeric=True)
            df_train_ling = utils_pipeline.get_ling_feats()
            df_train = pd.concat([df_train_log.reset_index(drop=True), df_train_ling.reset_index(drop=True)],
                                 axis='columns')
        else:
            print("Loading only logging features...")
            df_train = utils_pipeline.load_dataframe(split=Split.TRAIN, only_numeric=True)
        X = df_train.to_numpy()
        y = utils_pipeline.get_labels(Split.TRAIN)
        # import pdb; pdb.set_trace()
        y_encoder = utils_pipeline.get_label_encoder(Split.TRAIN)

        if not args.grid:
            train_kfold(model, X, y, y_encoder)
        else:
            gridsearch_cv(model, X, y)
