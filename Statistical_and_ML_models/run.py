"""Script to run Statistical and ML model experiments."""

import argparse
import logging
import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from util import LingFeatDF


def select_columns(dataframe: pd.DataFrame, dtype_include: list, name_exclude: list) -> pd.DataFrame:
    """Return df with only those columns that have a certain datatype and are not in a list of undesired columns."""
    return dataframe[
        [col for col in dataframe.columns if dataframe[col].dtype in dtype_include and col not in name_exclude]
    ]


def threshold_regression_prediction(predictions: np.ndarray) -> np.ndarray:
    """Convert softmax regression values within a vector to one-hot encoding, based on argmax."""
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
    return np.asarray(preds_one_hot)


def do_kfold_scoring(model: Union[LinearRegression, RandomForestClassifier],
                     X: pd.DataFrame, y: np.ndarray, selector: SelectKBest = None, scaling=True) -> None:
    """Perform a k-fold CV with given model on the supplied dataset."""
    if selector:
        X = selector.transform(X)
    else:
        X = X.to_numpy()
    # Scaling has to be performed individually for each run, because the number of features may be different per
    # each experiment, and the scaler requires same amount to transform once 'fitted'
    if scaling:
        X = StandardScaler().fit_transform(X)

    scores_train: List[float] = []
    scores_valid: List[float] = []
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = label_encoder.transform(y[train_index]), label_encoder.transform(y[valid_index])
        model.fit(X_train, y_train)
        if model._estimator_type == 'regressor':
            scores_train.append(accuracy_score(threshold_regression_prediction(model.predict(X_train)), y_train))
            scores_valid.append(accuracy_score(threshold_regression_prediction(model.predict(X_valid)), y_valid))
        else:
            scores_train.append(model.score(X_train, y_train))
            scores_valid.append(model.score(X_valid, y_valid))
    print("Average train score:", round(float(np.mean(scores_train)), 4))
    print("Average validation score:", round(np.mean(scores_valid), 4))
    print("Standard deviation of validation:", round(np.std(scores_valid), 4))


def get_features_sorted(selector: SelectKBest) -> List[Tuple[str, float]]:
    """Return sorted features and their relevance according to a KBest selector."""
    return sorted(zip(list(selector.get_feature_names_out()), list(selector.scores_)),
                  key=lambda x: x[1], reverse=True)


def run_numeric_data_experiments(model: Union[LinearRegression, RandomForestClassifier],
                                 X_train_numeric: pd.DataFrame, y: np.ndarray, scaling=False) -> None:
    """Run three type of experiments: [Using KBest features, using Keystroke features, using post-edit features]."""
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
    """Remove features that are not present in all dataframes."""
    return list(set(lst1) & set(lst2))


def intersect_linguistic_columns(train_mt: pd.DataFrame, train_tgt: pd.DataFrame,
                                 test_mt: pd.DataFrame, test_tgt: pd.DataFrame) -> list:
    """Merge columns of the extracted linguistic feature dataframes, after having used the linguistic tool API."""
    all_columns = [list(train_mt.columns), list(train_tgt.columns), list(test_mt.columns), list(test_tgt.columns)]
    # initiate intersected columns with an arbitrary dataframes' columns
    intersected_columns = random.choice(all_columns)
    for cols in all_columns:
        intersected_columns = intersection(intersected_columns, cols)
    return intersected_columns


def filter_linguistic_columns(df_ling: pd.DataFrame, intersected_columns: list) -> pd.DataFrame:
    """Return only columns of the linguistic dataframe that are found within the 'intersected_columns' list."""
    if len(intersection(list(df_ling.columns), intersected_columns)) < len(intersected_columns):
        raise ValueError("Unfiltered linguistic dataframe has missing columns. Dataframe columns must include at least "
                         "all elements of 'intersected_columns' argument.")
    return df_ling[[intersected_columns]]


def subtract_df(df_mt: LingFeatDF, df_tgt: LingFeatDF) -> pd.DataFrame:
    """Subtract machine-translated dataframe of ling features from post-edited dataframe of ling features."""
    if isinstance(df_mt, LingFeatDF) and isinstance(df_tgt, LingFeatDF):
        if df_mt.name.split('_')[0] != df_tgt.name.split('_')[0]:
            raise SyntaxError(
                "Wrong values of LingFeatDF have been given, 'name' attributes must have identical beginning."
            )
        if not df_mt.name.endswith('mt') and not df_tgt.name.endswith('tgt'):
            raise SyntaxError("Dataframes have been given in the wrong order. 'mt' is required to be first.")
    else:
        raise TypeError("Subtraction of DataFrames only possible between LingFeatDF namedtuples.")
    return df_tgt.df.subtract(df_mt.df)


# TODO: Get feature names and fix the top20 feature selection
def get_feature_importances(rf_model: RandomForestClassifier):
    """Retrieve sorted feature importances of RandomForestClassifier."""
    importances = sorted(rf_model.feature_importances_, reverse=True)
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)[0:20]
    forest_importances = pd.Series(importances).iloc[0:20]
    return forest_importances, std


def save_feature_importance_plots(forest_importances: pd.Series, std: float, experiment_type: str) -> None:
    """Plot and save feature importances of RandomForest model."""
    fig, ax = plt.subplots()
    # fig.tight_layout(h_pad=10)
    forest_importances.plot.barh(yerr=std, ax=ax)
    ax.set_title('MDI-based Feature Importances')
    ax.set_ylabel("Features")
    ax.set_xlabel("Mean decrease in impurity")
    fig.savefig(f"../images/RandomForest_{experiment_type}_importances.jpg")


def run_test(model, X_train, y_train, X_test, experiment_type: str) -> None:
    """Run model on test set and obtain predictions, but train the model first."""
    model_fit = model.fit(X_train, y_train)
    if model._estimator_type == 'regressor':
        predictions = transform_predictions(threshold_regression_prediction(model_fit.predict(X_test)))
    else:
        predictions = transform_predictions(model_fit.predict(X_test))
    output_test_predictions(model_fit, predictions, experiment_type)


def transform_predictions(predictions: np.ndarray) -> str:
    """Transform one-hot encoded labels into integer labels."""
    transformed_predictions = []
    for vector in predictions:
        transformed_predictions.append(str(np.argmax(vector)))
    return ''.join(transformed_predictions)


def output_test_predictions(model: Union[LinearRegression, RandomForestClassifier],
                            predictions: Union[np.array, list], experiment_type: str) -> None:
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
                        help="The type of sklearn model to run the experiments with."
                             " Can either be LinearRegression (lr) or RandomForestClassifier (rf)")
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
    df_train: pd.DataFrame = load_dataset("GroNLP/ik-nlp-22_pestyle", "full",
                                          data_dir='../IK_NLP_22_PESTYLE')['train'].to_pandas()
    df_test: pd.DataFrame = load_dataset("GroNLP/ik-nlp-22_pestyle", "mask_subject",
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
    logger.info("Loading linguistic features...")

    lingfeat = [
        LingFeatDF(pd.read_csv('../Linguistic_features/train_mt.csv', sep="\t").drop(columns=['Filename']), 'train_mt'),
        LingFeatDF(pd.read_csv('../Linguistic_features/test_mt.csv', sep='\t').drop(columns=['Filename']), 'test_mt'),
        LingFeatDF(pd.read_csv('../Linguistic_features/train_tgt.csv', sep="\t").drop(columns=['Filename']),
                   'train_tgt'),
        LingFeatDF(pd.read_csv('../Linguistic_features/test_tgt.csv', sep="\t").drop(columns=['Filename']), 'test_tgt')
    ]
    intersected_linguistic_columns = intersect_linguistic_columns(*[nt.df for nt in lingfeat])
    col_lens = []
    for idx, nt in enumerate(lingfeat):
        lingfeat[idx] = filter_linguistic_columns(nt.df, intersected_linguistic_columns)
        col_lens.append(lingfeat[idx].df.columns)
    assert len(set(col_lens)) == 1, "Number of columns have to be equal for all linguistic feature dataframes."

    X_train_ling = subtract_df(lingfeat[0], lingfeat[2])
    X_test_ling = subtract_df(lingfeat[1], lingfeat[3])
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
