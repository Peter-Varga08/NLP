from datasets import load_dataset, list_metrics, load_metric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from typing import Union, List, Tuple
import argparse
import os
import logging


def select_columns(dataframe: pd.DataFrame, dtype_include: list, name_exclude: list) -> pd.DataFrame:
    """Return a dataframe with only those columns that have a certain datatype and are not in a list of undesired
    columns."""
    df_selected = dataframe[
        [col for col in dataframe.columns if dataframe[col].dtype in dtype_include and col not in name_exclude]]
    return df_selected


def threshold_regression_prediction(predictions: np.array) -> np.array:
    """Converts softmax regression values within a vector to one-hot encoding, based on argmax."""
    preds_one_hot = []
    for vector in predictions:
        argmax = np.argmax(vector)
        tmp = []
        for idx, pred in enumerate(vector):
            if idx == argmax:
                tmp.append(1)
            else:
                tmp.append(0)
        preds_one_hot.append(tmp)
    preds_one_hot = np.array(preds_one_hot)
    return preds_one_hot


def do_kfold_scoring(model: Union[LinearRegression, RandomForestClassifier],
                     X: pd.DataFrame, y: np.array, selector: SelectKBest = None, scaling=True) -> None:
    """Performs a k-fold CV with given model on the supplied dataset"""
    if selector:
        X = selector.transform(X)
    else:
        X = X.to_numpy()
    # Scaling has to be performed individually for each run, because the number of features may be different per
    # each experiment, and the scaler requires same amount to transform once 'fitted'
    if scaling:
        X = StandardScaler().fit_transform(X)

    scores_train = []
    scores_valid = []
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = label_encoder.transform(y[train_index]), label_encoder.transform(y[valid_index])
        model_fit = model.fit(X_train, y_train)
        if model._estimator_type == 'regressor':
            scores_train.append(accuracy_score(threshold_regression_prediction(model_fit.predict(X_train)), y_train))
            scores_valid.append(accuracy_score(threshold_regression_prediction(model_fit.predict(X_valid)), y_valid))
        else:
            scores_train.append(model_fit.score(X_train, y_train))
            scores_valid.append(model_fit.score(X_valid, y_valid))
    print("Average train score:", round(np.mean(scores_train), 4))
    print("Average validation score:", round(np.mean(scores_valid), 4))
    print("Standard deviation of validation:", round(np.std(scores_valid), 4))


def get_features_sorted(selector: SelectKBest) -> List[Tuple[str, float]]:
    """Returns a list of tuples, containing the name of the features and their relevance according to a KBest selector"""
    features_sorted = sorted(zip(list(selector.get_feature_names_out()), list(selector.scores_)),
                             key=lambda x: x[1], reverse=True)
    return features_sorted


def run_numeric_data_experiments(model: Union[LinearRegression, RandomForestClassifier],
                                 X_train_numeric: np.array, y: np.array, scaling=False) -> None:
    """Runs three type of experiments: [Using KBest features, using Keystroke features, using post-edit features]"""

    print(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{model}]...\n{'-' * 50}")
    # Train and validate on TOP 100%, 75%, 50% and 25% features of the data
    for ratio in [1, 0.75, 0.5, 0.25]:
        kbest_ = SelectKBest(chi2, k=int(len(X_train_numeric.columns) * ratio)).fit(X_train_numeric, y)
        print(f"Performing 10-Fold CV on top {int(ratio * 100)}% features of the data...")
        do_kfold_scoring(model, X_train_numeric, y, selector=kbest_, scaling=scaling)
        print("*" * 40, "\n")

    # Train and validate on keystroke features data
    print(f"\n{'*' * 40}\n[KEYSTROKE FEATURES]")
    keystroke_columns = [col for col in X_train_numeric.columns if col.startswith('k_')]
    do_kfold_scoring(model, X_train_numeric[keystroke_columns], y, scaling=scaling)

    # Train and validate on postedit features data
    print(f"\n{'*' * 40}\n[POSTEDIT FEATURES]")
    postedit_columns = [col for col in X_train_numeric.columns if col.startswith('n_')]
    do_kfold_scoring(model, X_train_numeric[postedit_columns], y, scaling=scaling)


def intersection(lst1: list, lst2: list) -> list:
    """Used for removing features that are not present in all dataframes"""
    return list(set(lst1) & set(lst2))


def filter_linguistic_columns(train_mt, train_tgt, test_mt, test_tgt):
    all_columns = [list(train_mt.columns), list(train_tgt.columns), list(test_mt.columns), list(test_tgt.columns)]
    merged_columns = list(train_mt.columns)
    for cols in all_columns:
        merged_columns = intersection(merged_columns, cols)
    train_mt = train_mt[[col for col in train_mt.columns if col in merged_columns]]
    train_tgt = train_tgt[[col for col in train_tgt.columns if col in merged_columns]]
    test_mt = test_mt[[col for col in test_mt.columns if col in merged_columns]]
    test_tgt = test_tgt[[col for col in test_tgt.columns if col in merged_columns]]

    assert len(train_mt.columns) == len(train_tgt.columns) == len(test_mt.columns) == len(test_tgt.columns)

    X_train_ling = train_tgt.subtract(train_mt)
    X_test_ling = test_tgt.subtract(test_mt)

    return X_train_ling, X_test_ling

# TODO: Get feature names and fix the top20 feature selection
def get_feature_importances(rf_model: RandomForestClassifier):
    importances = sorted(rf_model.feature_importances_, reverse=True)
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)[0:20]
    forest_importances = pd.Series(importances).iloc[0:20]
    return forest_importances, std


def save_feature_importance_plots(forest_importances: pd.Series, std: float, experiment_type: str):
    fig, ax = plt.subplots()
    # fig.tight_layout(h_pad=10)
    forest_importances.plot.barh(yerr=std, ax=ax)
    ax.set_title('MDI-based Feature Importances')
    ax.set_ylabel("Features")
    ax.set_xlabel("Mean decrease in impurity")
    fig.savefig(f"../images/RandomForest_{experiment_type}_importances.jpg")


def run_test(model, X_train, y_train, X_test, experiment_type: str):
    model_fit = model.fit(X_train, y_train)
    if model._estimator_type == 'regressor':
        predictions = transform_predictions(threshold_regression_prediction(model_fit.predict(X_test)))
    else:
        predictions = transform_predictions(model_fit.predict(X_test))
    output_test_predictions(model_fit, predictions, experiment_type)


def transform_predictions(predictions: np.array) -> str:
    """Transforms one-hot encoded labels into integer labels"""
    transformed_predictions = []
    for vector in predictions:
        transformed_predictions.append(str(np.argmax(vector)))
    return ''.join(transformed_predictions)


def output_test_predictions(model: Union[LinearRegression, RandomForestClassifier],
                            predictions: Union[np.array, list], experiment_type: str) -> np.array:
    """Generate txt file that contains all the predictions in a sequence within a single row."""
    with open(f"predictions/{model._estimator_type}_{experiment_type}_predictions.txt", 'w') as f:
        f.write(predictions)


if __name__ == "__main__":
    # |-------------------------|
    # |       Parse args        |
    # |-------------------------|
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-e', '--experiment_type', type=str, default='numeric',
                        choices=['numeric', 'linguistic', 'combined'],
                        help='Type of experiment to run, w.r.t. the features being used.')
    parser.add_argument('-m', '--model_type', type=str, default=None, required=True, choices=['lr', 'rf'],
                        help="The type of sklearn model to run the experiments with. Can either be LinearRegression (lr)" \
                             "or RandomForestClassifier (rf)")
    parser.add_argument('-s', '--scaling', action='store_true', help='Whether to use scaling on features or not.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to enable or disable logging.')
    parser.add_argument('-t', '--test', action='store_true', help='Whether to run tests or not.')
    parser.add_argument('-p', '--plots', action='store_true', help='Whether to create plots or not')
    args = parser.parse_args()

    logger = logging.getLogger()
    if not args.verbose:
        logger.disabled = True

    # |-------------------------|
    # |     Load datasets       |
    # |-------------------------|
    logger.info("Loading datasets...")
    df_train = load_dataset("GroNLP/ik-nlp-22_pestyle", "full",
                            data_dir='../IK_NLP_22_PESTYLE')['train'].to_pandas()
    df_test = load_dataset("GroNLP/ik-nlp-22_pestyle", "mask_subject",
                           data_dir='../IK_NLP_22_PESTYLE')['test'].to_pandas()
    df_train = df_train[df_train.modality != 'ht']
    df_test = df_test[df_test.modality != 'ht']
    y_train = np.array(df_train.subject_id)
    y_test = pd.read_csv('test_ground_truths.csv', header=None).to_numpy()
    # one label-encoder is enough, categories are the same for both datasets
    assert np.array_equal(np.unique(y_train),
                          np.unique(y_test)), "Unique labels within train and test set are different."
    label_encoder = LabelBinarizer().fit(y_train)
    y_train_encoded = label_encoder.transform(y_train)  # used for 'run_test()' only

    # Numeric features
    logger.info("Creating numeric features.")
    X_train_numeric = select_columns(df_train, dtype_include=['float32', 'int32'], name_exclude=['item_id'])
    X_test_numeric = select_columns(df_test, dtype_include=['float32', 'int32'], name_exclude=['item_id'])
    # Linguistic features
    logger.info("Loading linguistic features")
    lingfeat_train_mt = pd.read_csv('../Linguistic_features/train_mt.csv', sep="\t").drop(
        columns=['Filename'])
    lingfeat_test_mt = pd.read_csv('../Linguistic_features/test_mt.csv', sep='\t').drop(
        columns=['Filename'])
    lingfeat_train_tgt = pd.read_csv('../Linguistic_features/train_tgt.csv', sep="\t").drop(
        columns=['Filename'])
    lingfeat_test_tgt = pd.read_csv('../Linguistic_features/test_tgt.csv', sep="\t").drop(
        columns=['Filename'])
    X_train_ling, X_test_ling = filter_linguistic_columns(lingfeat_train_mt, lingfeat_train_tgt,
                                                          lingfeat_test_mt, lingfeat_test_tgt)
    # Combined features
    X_train_combined = pd.concat([X_train_numeric.reset_index(drop=True), X_train_ling.reset_index(drop=True)], axis=1)
    X_test_combined = pd.concat([X_test_numeric.reset_index(drop=True), X_test_ling.reset_index(drop=True)], axis=1)

    # |-------------------------|
    # |       Load models       |
    # |-------------------------|
    if args.model_type == 'lr':
        estimator = LinearRegression(n_jobs=-1)
    elif args.model_type == 'rf':
        estimator = RandomForestClassifier(n_jobs=-1, random_state=42)
    else:
        raise ValueError('Invalid model type.')

    # |-------------------------|
    # |     Run experiments     |
    # |-------------------------|
    if args.experiment_type == 'numeric':
        if not args.test:
            run_numeric_data_experiments(estimator, X_train_numeric, y_train, scaling=args.scaling)
            estimator.fit(X_train_numeric, y_train)  # post-fitting model to all train data for feature importance
        else:
            run_test(estimator, X_train_numeric, y_train_encoded, X_test_numeric, args.experiment_type)
    elif args.experiment_type == 'linguistic':
        if not args.test:
            logger.info(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{estimator}]...\n{'-' * 50}")
            do_kfold_scoring(estimator, X_train_ling, y_train, scaling=args.scaling)
            logger.info("*" * 40, "\n")
            estimator.fit(X_train_ling, y_train_encoded)
        else:
            run_test(estimator, X_train_ling, y_train_encoded, X_test_ling, args.experiment_type)
    elif args.experiment_type == 'combined':
        if not args.test:
            logger.info(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{estimator}]...\n{'-' * 50}")
            do_kfold_scoring(estimator, X_train_combined, y_train, scaling=args.scaling)
            logger.info("*" * 40, "\n")
            estimator.fit(X_train_combined, y_train_encoded)
        else:
            run_test(estimator, X_train_combined, y_train_encoded, X_test_combined, args.experiment_type)
    else:
        raise ValueError("Invalid experiment type.")

    # Plot feature importance
    if args.model_type == "rf" and args.plots:
        importances, std = get_feature_importances(estimator)
        save_feature_importance_plots(importances, std, args.experiment_type)
