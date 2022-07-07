"""
- Training script for all models included in the paper:
'Comparing Statistical and Machine Learning Approaches to Identify Post-Editors from Logging and Linguistics Features:
 an Investigative Stylometry Study'.

- Example usage for training a RandomForest classifier model with base params: 'python3 train.py ML_CLF_RF'
"""

import argparse
import pdb
import warnings
from typing import List, Union, Dict, Tuple

import numpy as np
import pandas as pd
import wandb
from numpy.typing import NDArray
from scikeras.wrappers import KerasClassifier
from simpletransformers.config.model_args import ClassificationArgs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

from src.enums import MetricType, Split
from src.model import ClassificationTransformer, NeuralNetwork, get_model_classname
from src.utils import wandb_, metrics, pipeline
from src.utils.logger import LOGGER
from src.utils.pipeline import softmax2onehot

MLModel = Union[
    RandomForestClassifier,
    LinearSVC,
    NeuralNetwork,
    KerasClassifier,
    LogisticRegression,
    GridSearchCV,
]
SPLITS = 10


def train_kfold(
    model: MLModel,
    X: NDArray,
    y: NDArray,
    splits: int = 10,
):
    """Train a model using logging/linguistic features with k-fold split."""
    clf_report: List
    accuracy: List[float]

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    label_encoder = LabelBinarizer().fit(y)
    clf_report = []
    accuracy = []
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        y_train_encoded, y_valid_encoded = label_encoder.transform(
            y_train
        ), label_encoder.transform(y_valid)
        X_train_scaled, X_valid_scaled = pipeline.scaling(X_train, X_valid)

        model_name = get_model_classname(model)
        # For the neural network, fit function requires validation data to do early-stopping
        if model_name == "NeuralNetwork":
            model.fit(X_train_scaled, y_train_encoded, X_valid_scaled, y_valid_encoded)
            y_preds = softmax2onehot(model.predict(X_valid_scaled))
        else:
            model = model.fit(X_train_scaled, y_train_encoded)
            y_preds = model.predict(X_valid_scaled)

        clf_report.append(precision_recall_fscore_support(y_valid_encoded, y_preds))
        accuracy.append(accuracy_score(y_valid_encoded, y_preds))

    result = {
        MetricType.CLF_REPORT: clf_report,
        MetricType.ACCURACY_SCORE: accuracy,
    }

    return result


def train_bert(model: ClassificationTransformer, data: pd.DataFrame):
    pass


def train_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # |-----------------------------------------|
    # |            BASE ARGUMENTS               |
    # |-----------------------------------------|
    parser.add_argument(
        "-tr",
        "--train_file",
        default="../data/",
        type=str,
        help="Location of training file.",
    )
    parser.add_argument(
        "-f",
        "--features",
        help="Which features to include.",
        choices=["log", "ling", "both"],
        default="log",
    )
    parser.add_argument(
        "--n_jobs",
        help="Number of cpu cores to use.",
        type=int,
        default=-1,  # use all
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--encoder", action="store_true")
    # |-----------------------------------------|
    # |            BERT MODEL                   |
    # |-----------------------------------------|
    bert_parser = subparsers.add_parser("BERT")
    bert_parser.add_argument("-subparser", "--subparser_bert", default=True)
    bert_parser.add_argument(
        "-c",
        "--bert_config",
        type=str,
        default="dbmdz/bert-base-italian-uncased",
        help="Specify a model configuration from HuggingFace.",
    )
    # |-----------------------------------------|
    # |            NEURAL NETWORK               |
    # |-----------------------------------------|
    # Specified parameters based training
    nn_parser = subparsers.add_parser("NN")
    nn_parser.add_argument("-subparser", "--subparser_nn", default=True)
    nn_parser.add_argument("--lr", type=float, default=1e-3)
    nn_parser.add_argument("--batch_size", type=int, default=16)
    nn_parser.add_argument("--epochs", type=int, default=1000)
    nn_parser.add_argument("--dropout", type=float, default=0.1)
    nn_parser.add_argument("--clipvalue", type=float, default=0.5)
    nn_parser.add_argument("--optimizer", type=str, default="Adam")
    nn_parser.add_argument("--patience", type=int, default=100)

    # |-----------------------------------------|
    # |           CLASSIFIER ML MODELS          |
    # |-----------------------------------------|
    # 1) LinearSVC
    svc_parser = subparsers.add_parser("ML_CLF_SVC")
    svc_parser.add_argument("-subparser", "--subparser_ml_clf_svc", default=True)
    svc_parser.add_argument("--penalty", type=str, choices=["l1", "l2"], default="l1")
    svc_parser.add_argument(
        "--loss", type=str, choices=["hinge", "squared_hinge"], default="squared_hinge"
    )
    svc_parser.add_argument("-d", "--dual", action="store_true")
    svc_parser.add_argument(
        "--C",
        type=float,
        help="Inversely proportional regularization parameter.",
        default=1.0,
    )
    svc_parser.add_argument("--max_iter", type=int, default=1000)
    svc_parser.add_argument("--tol", type=float, default=1e-5)

    # 2) RandomForest
    randomforest_parser = subparsers.add_parser("ML_CLF_RF")
    randomforest_parser.add_argument(
        "-subparser", "--subparser_ml_clf_rf", default=True
    )
    randomforest_parser.add_argument(
        "-ne",
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators in the forest.",
    )
    randomforest_parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="None means expansion until all leaves are pure,"
        " or contain less than min_samples",
    )
    randomforest_parser.add_argument("--min_samples", type=int, default=2)

    # 3) LogisticRegression
    regressor_parser = subparsers.add_parser("ML_REG")
    regressor_parser.add_argument("-subparser", "--subparser_ml_reg", default=True)
    regressor_parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Subtract mean and divide by L2-norm.",
    )
    regressor_parser.add_argument(
        "--penalty", type=str, default="elasticnet"
    )  # elasticnet adds both l1 and l2 terms
    regressor_parser.add_argument("--tol", type=float, default=1e4)
    regressor_parser.add_argument("--C", type=float, default=1)
    regressor_parser.add_argument("--l1_ratio", type=float, default=0)
    regressor_parser.add_argument("--max_iter", type=float, default=1000)
    regressor_parser.add_argument("--solver", type=str, default="saga")

    return parser.parse_args()


def load_model(
    model_args: argparse.Namespace,
    param_grid: Dict[str, Dict[str, List[Union[float, int, str]]]],
) -> Union[MLModel, ClassificationTransformer]:
    """Select a specific classifier model based on parsed arguments."""
    cv = ShuffleSplit(
        n_splits=10, test_size=0.1
    )  # use same CV for all gridsearch models
    # |--------------------|
    # |       BERT         |
    # |--------------------|
    if hasattr(model_args, "subparser_bert"):
        # Define model
        bert_args = ClassificationArgs()
        bert_args.num_train_epochs = 3
        bert_args.overwrite_output_dir = True
        bert_args.learning_rate = 0.000001
        bert_args.do_lower_case = True
        bert_args.silence = True
        model = ClassificationTransformer(model_args.bert_config, bert_args)
    else:
        # |--------------------|
        # |   Neural Network   |
        # |--------------------|
        if hasattr(model_args, "subparser_nn"):
            if not model_args.grid:
                model = NeuralNetwork(
                    input_len=model_args.input_len,
                    lr=model_args.lr,
                    clipvalue=model_args.clipvalue,
                    patience=model_args.patience,
                    batch_size=model_args.batch_size,
                    epochs=model_args.epochs,
                    optimizer=args.optimizer,
                )
            else:
                # TODO: FIX KerasClassifier params, e.g. "Invalid parameter clipvalue for estimator KerasClassifier."
                model = GridSearchCV(
                    estimator=KerasClassifier(
                        NeuralNetwork(
                            input_len=model_args.input_len,
                            output_len=model_args.output_len,
                        ).model
                    ),
                    param_grid=param_grid["KerasClassifier"],
                    cv=cv,
                    scoring="accuracy",
                    verbose=1,
                    return_train_score=True,
                )
        # |--------------------|
        # |     REGRESSORS     |
        # |--------------------|
        elif hasattr(model_args, "subparser_ml_reg"):
            if not model_args.grid:
                model = LogisticRegression(
                    n_jobs=model_args.n_jobs,
                    l1_ratio=model_args.l1_ratio,
                    penalty=model_args.penalty,
                    max_iter=model_args.max_iter,
                    tol=model_args.tol,
                    random_state=model_args.random_state,
                    solver=model_args.solver,
                    verbose=1,
                )
            else:
                model = GridSearchCV(
                    estimator=LogisticRegression(
                        penalty="elasticnet", solver=model_args.solver
                    ),
                    param_grid=param_grid["LogisticRegression"],
                    cv=cv,
                    n_jobs=model_args.n_jobs,
                    scoring="accuracy",
                    verbose=1,
                    return_train_score=True,
                )
        # |--------------------|
        # |     LinearSVC      |
        # |--------------------|
        elif hasattr(model_args, "subparser_ml_clf_svc"):
            if not model_args.grid:
                model = LinearSVC(
                    C=model_args.C,
                    tol=model_args.tol,
                    loss=model_args.loss,
                    penalty=model_args.penalty,
                    max_iter=model_args.max_iter,
                    dual=model_args.dual,
                    verbose=1,
                )
            else:
                model = GridSearchCV(
                    estimator=LinearSVC(),
                    param_grid=param_grid["LinearSVC"],
                    cv=cv,
                    n_jobs=-model_args.n_jobs,
                    scoring="accuracy",
                    verbose=1,
                    return_train_score=True,
                )
        # |--------------------|
        # |   RANDOMFOREST CLF |
        # |--------------------|
        elif hasattr(model_args, "subparser_ml_clf_rf"):
            if not model_args.grid:
                model = RandomForestClassifier(
                    n_estimators=model_args.n_estimators,
                    max_depth=model_args.max_depth,
                    min_samples_split=model_args.min_samples,
                    random_state=42,
                    n_jobs=-1,
                    verbose=1,
                )
            else:
                model = GridSearchCV(
                    estimator=RandomForestClassifier(n_jobs=-1),
                    param_grid=param_grid["RandomForestClassifier"],
                    cv=cv,
                    scoring="accuracy",
                    verbose=1,
                    return_train_score=True,
                )
        else:
            raise SyntaxError(
                "In order to perform training, specification of subparser is required."
            )
    return model


def load_ml_train_dataset(dataset_choice: str = None) -> Tuple[NDArray, NDArray]:
    """Select the correct TRAINING dataset for a non-transformer ML model."""
    LOGGER.info("Preparing dataset for training Machine Learning model...")
    if dataset_choice == "ling":
        LOGGER.info("Loading linguistic features only...")
        df_train = pipeline.get_ling_feats(Split.TRAIN)
    elif dataset_choice == "both":
        LOGGER.info("Loading linguistic and logging features...")
        df_train_log = pipeline.load_dataframe(split=Split.TRAIN, only_numeric=True)
        df_train_ling = pipeline.get_ling_feats()
        df_train = pd.concat(
            [
                df_train_log.reset_index(drop=True),
                df_train_ling.reset_index(drop=True),
            ],
            axis="columns",
        )
    else:
        LOGGER.info("Loading only logging features...")
        df_train = pipeline.load_dataframe(split=Split.TRAIN, only_numeric=True)
    X = df_train.to_numpy()
    y = pipeline.get_labels(Split.TRAIN)
    return X, y


if __name__ == "__main__":
    LOGGER.info("Parsing arguments...")
    warnings.filterwarnings("ignore")
    args = train_parser()

    gridsearchcv_param_grid = {
        "KerasClassifier": {
            "batch_size": [16, 32, 64],
            "dropout": [0.1, 0.2, 0.3],
            "optimizer": ["Adam", "RMSprop"],
            "patience": [20],
            "clipvalue": [0.25, 0.5, 0.75, 1],
            "lr": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5],
        },
        "LogisticRegression": {
            "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "max_iter": [100, 250, 500, 1000, 2000],
            "tol": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        },
        "RandomForestClassifier": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 5, 10, 15],
            "min_samples_split": [2, 3, 4, 5, 6, 7],
        },
        "LinearSVC": {
            "C": [1, 5, 10, 25, 50, 100],
            "tol": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
            "loss": ["hinge", "squared_hinge"],
            "penalty": ["l1", "l2"],
            "max_iter": [100, 250, 500, 1000],
        },
    }

    if not hasattr(args, "subparser_bert"):
        # |-------------------|
        # |    Load DATASET   |
        # |-------------------|
        X_train, y_train = load_ml_train_dataset(args.features)
        # |-----------------------------|
        # |    Load and Train MODEL     |
        # |-----------------------------|
        vargs = vars(args)
        vargs.update(
            {"input_len": X_train.shape[1], "output_len": len(np.unique(y_train))}
        )
        args = argparse.Namespace(**vargs)
        estimator = load_model(args, gridsearchcv_param_grid)

        LOGGER.info("Training Machine Learning model...")
        if not args.grid:
            estimator_name = get_model_classname(estimator)
            wandb.init(project="NLP", entity="petervarga", name=estimator_name)
            wandb.config.update(args)

            scores = train_kfold(estimator, X_train, y_train, splits=SPLITS)
            # pdb.set_trace()
            score = metrics.get_avg_score(scores)
            # LOG TO WANDB
            score[MetricType.CLF_REPORT] = metrics.explain_clf_score(
                score[MetricType.CLF_REPORT]
            )
            LOGGER.info(
                f"Logging 10-FOLD scores for model [{estimator_name}] onto WANDB..."
            )
            wandb.log(score)
            LOGGER.info(
                f"Logging ClassificationReport for model [{estimator_name}] onto WANDB..."
            )
            wandb.log(
                {
                    estimator_name: wandb_.create_clf_report_table(
                        estimator_name, score[MetricType.CLF_REPORT]
                    )
                }
            )
            # TODO 4: Add feature importance implementation and logging/saving.
            # plot_feature_importances(estimator, df_train.columns)
            # pp(score)
        else:
            grid_name = get_model_classname(estimator)
            estimator_name = get_model_classname(estimator.estimator)
            wandb.init(
                project="NLP", entity="petervarga", name=f"{grid_name}_{estimator_name}"
            )
            wandb.config.update(args)
            wandb.config.update(
                {"param_grid": gridsearchcv_param_grid[estimator_name]},
                allow_val_change=True,
            )

            X_train_scaled = pipeline.scaling(X_train)
            pdb.set_trace()
            estimator.fit(X_train_scaled, y_train)
            LOGGER.info(
                f"Logging GridSearchCV results for model [{estimator_name}] onto WANDB..."
            )
            wandb.run.summary.update(
                {
                    f"{estimator_name}": {
                        "best_cv_validation_score": np.nanmax(
                            estimator.cv_results_["mean_test_score"]
                        ),
                        "best_cv_training_score": np.nanmax(
                            estimator.cv_results_["mean_train_score"]
                        ),
                        "best_params_": estimator.best_params_,
                        "best_estimator_": str(estimator.best_estimator_),
                    }
                }
            )
            wandb.run.summary["n_features"] = estimator.n_features_in_
            wandb.run.summary["n_splits"] = estimator.n_splits_
            wandb.run.summary["scoring"] = estimator.scoring
    else:
        pass  # TODO 5: Bert training
        estimator = load_model(args, gridsearchcv_param_grid)
        X = None
        y = None
        LOGGER.info("Training BERT model...")
        # estimator = estimator.fit()
