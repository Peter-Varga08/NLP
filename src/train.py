"""
Example usage: python train.py ML_CLF_RF
"""

import argparse
import pprint
from typing import List, Union, Dict

import pandas as pd
from numpy.typing import NDArray
from scikeras.wrappers import KerasClassifier
from simpletransformers.config.model_args import ClassificationArgs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

from enums import MetricType, Split
from model import ClassificationTransformer, NeuralNetwork
from utils import metrics, pipeline

MLModel = Union[RandomForestClassifier, LinearSVC, KerasClassifier, LogisticRegression]
SPLITS = 10


def train_kfold(
    model: MLModel,
    X: NDArray,
    y: NDArray,
    encoder: LabelBinarizer = None,
    splits: int = 10,
):
    """Train a model using logging/linguistic features with k-fold split."""
    clf_report: List
    accuracy: List[float]

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    clf_report = []
    accuracy = []
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        if encoder:
            y_train, y_valid = encoder.transform(
                y_train,
            ), encoder.transform(y_valid)
        X_train_scaled, X_valid_scaled = pipeline.scaling(X_train, X_valid)
        model = model.fit(X_train_scaled, y_train)
        y_preds = model.predict(X_valid_scaled)

        # if model._estimator_type == "regressor":
        #     y_preds = threshold_regression_prediction(y_preds)
        clf_report.append(precision_recall_fscore_support(y_valid, y_preds))
        accuracy.append(accuracy_score(y_valid, y_preds))

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
        help="Which features to include",
        choices=["log", "ling", "both"],
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to enable or disable logging.",
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
    nn_parser.add_argument(
        "-es",
        "--early_stopping",
        type=int,
        help="Patience of early stopping.",
        default=100,
    )
    nn_parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
    nn_parser.add_argument("-b", "--batch_size", type=int, default=16)
    nn_parser.add_argument("-ep", "--epochs", type=int, default=50)
    nn_parser.add_argument("-d", "--dropout", type=float, default=0.1)
    nn_parser.add_argument("-c", "--clipvalue", type=float, default=0.5)
    nn_parser.add_argument("-o", "--optimizer", type=str, default="Adam")

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


def select_model(
    args: argparse.Namespace,
    param_grid: Dict[str, Dict[str, List[Union[float, int, str]]]],
) -> Union[MLModel, ClassificationTransformer]:
    # |--------------------|
    # |       BERT         |
    # |--------------------|
    if hasattr(args, "subparser_bert"):
        # Define model
        model_args = ClassificationArgs()
        model_args.num_train_epochs = 3
        model_args.overwrite_output_dir = True
        model_args.learning_rate = 0.000001
        model_args.do_lower_case = True
        model_args.silence = True
        model = ClassificationTransformer(args.bert_config, model_args)
    else:
        # |--------------------|
        # |   Neural Network   |
        # |--------------------|
        if hasattr(args, "subparser_nn"):
            if not args.grid:
                nn = NeuralNetwork(
                    input_len=args.input_len, lr=args.lr, clipvalue=args.clip_value
                )
                # TODO: fix
                model = KerasClassifier(
                    nn.model, batch_size=args.batch_size, epochs=args.epochs, verbose=1
                )
            else:
                # TODO: add GridSearchCV to KerasCLassifier
                model = None
        # |--------------------|
        # |     REGRESSORS     |
        # |--------------------|
        elif hasattr(args, "subparser_ml_reg"):
            if not args.grid:
                model = LogisticRegression(
                    n_jobs=args.n_jobs,
                    l1_ratio=args.l1_ratio,
                    penalty=args.penalty,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    random_state=args.random_state,
                    solver=args.solver,
                )
            else:
                model = GridSearchCV(
                    estimator=LogisticRegression(
                        penalty="elasticnet", solver=args.solver
                    ),
                    param_grid=param_grid["LogisticRegression"],
                    cv=cv,
                    n_jobs=args.n_jobs,
                    scoring="accuracy",
                )
        # |--------------------|
        # |     LinearSVC      |
        # |--------------------|
        elif hasattr(args, "subparser_ml_clf_svc"):
            if not args.grid:
                model = LinearSVC(
                    C=args.C,
                    tol=args.tol,
                    loss=args.loss,
                    penalty=args.penalty,
                    max_iter=args.max_iter,
                    dual=args.dual,
                )
            else:
                model = GridSearchCV(
                    estimator=LinearSVC(),
                    param_grid=param_grid["LinearSVC"],
                    cv=cv,
                    n_jobs=-args.n_jobs,
                    scoring="accuracy",
                )
        # |--------------------|
        # |   RANDOMFOREST CLF |
        # |--------------------|
        elif hasattr(args, "subparser_ml_clf_rf"):
            if not args.grid:
                model = RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_split=args.min_samples,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = GridSearchCV(
                    estimator=RandomForestClassifier(n_jobs=-1),
                    param_grid=param_grid["RandomForest"],
                    cv=cv,
                    scoring="accuracy",
                    verbose=1,
                )
        else:
            raise SyntaxError(
                "In order to perform training, specification of subparser is required."
            )
    return model


if __name__ == "__main__":

    # wandb.init(project="NLP", entity="petervarga", name="suhano nyilvesszo")
    args = train_parser()
    print(vars(args))
    # wandb.config.update(args)

    # TODO: NeuralNetwork
    gridsearchcv_param_grid = {
        "NeuralNetwork": {},
        "LogisticRegression": {
            "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "max_iter": [100, 250, 500, 1000, 2000],
            "tol": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 5, 10, 15],
            "min_samples_split": [2, 3, 4, 5, 6, 7],
        },
        "LinearSVC": {
            "C": [1, 1.5, 5, 10],
            "tol": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
            "loss": ["hinge", "squared_hinge"],
            "penalty": ["l1", "l2"],
            "max_iter": [100, 250, 500, 1000, 2000],
        },
    }

    cv = ShuffleSplit(n_splits=10, test_size=0.1)
    estimator = select_model(args, gridsearchcv_param_grid)

    if not isinstance(estimator, ClassificationTransformer):
        print("Preparing dataset for training Machine Learning model...")
        if args.features == "ling":
            print("Loading linguistic features only...")
            df_train = pipeline.get_ling_feats(Split.TRAIN)
        elif args.features == "both":
            print("Loading linguistic and logging features...")
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
            print("Loading only logging features...")
            df_train = pipeline.load_dataframe(split=Split.TRAIN, only_numeric=True)
        X = df_train.to_numpy()
        y = pipeline.get_labels(Split.TRAIN)

        # if args.encoder:
        #     y_encoder = pipeline.get_label_encoder(Split.TRAIN)
        # else:
        #     y_encoder = None

        print("Training Machine Learning model...")
        if not args.grid:
            scores = train_kfold(estimator, X, y, splits=SPLITS)
            score = metrics.get_avg_score(scores)
            # LOG TO WANDB
            score[MetricType.CLF_REPORT] = metrics.explain_clf_score(
                score[MetricType.CLF_REPORT]
            )
            # wandb.log(score)
            # estimator_name = str(estimator).upper()
            # wandb.log({estimator_name: wandb_.create_clf_report_table(estimator_name, score[MetricType.CLF_REPORT])})
            # plot_feature_importances(estimator, df_train.columns)
            pprint.pprint(score)
        else:
            X_train_scaled = pipeline.scaling(X)
            print(len(X_train_scaled), len(y))
            estimator.fit(X_train_scaled, y)
            # import pdb
            # pdb.set_trace()
            # print(sorted(estimator.cv_results_.keys()))
            print("ESTIMATOR BEST SCORE:", estimator.best_score_)
            print("ESTIMATOR BEST PARAMS:", estimator.best_params_)
            print("ESTIMATOR BEST ESTIMATOR:", estimator.best_estimator_)
            # TODO: Add WANDB logging for best_estimator_, best_score, best_params_
    else:
        pass  # TODO: Bert training
        print("Training BERT model...")
