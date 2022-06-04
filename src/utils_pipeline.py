import argparse
import os
import random
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from enums import ConfigMode, Modality, Split

LingFeatDF = namedtuple('LingFeatDF', ['df', 'name'])


class ModelParser:
    def __init__(self) -> None:
        pass

    # TODO: ADD WRAPPER FOR ARGUMENT VALIDATION IF NEEDED
    @staticmethod
    def parse_nn_model() -> argparse:
        parser = argparse.ArgumentParser(description="Neural Network models argument parser.")
        parser.add_argument("-r", "--result_path", help="Directory where to save results.")
        parser.add_argument("-o", "--oversampling", default=False, help="Whether to oversample training data.")
        parser.add_argument("-k", "--kfold", default=False, help="Whether to do K(=10)-fold cross-validation.")
        parser.add_argument("-f", "--features", help="Which features to include", choices=["log", "ling", "both"])
        try:
            args = parser.parse_args()
            args.features = args.features.lower()
        except Exception as e:
            print(e)
            parser.print_help()
            sys.exit(0)
        mkdirs(args.result_path)
        return args

    @staticmethod
    def parse_ml_model() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Machine Learning models argument parser.")
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
        return parser.parse_args()

    @staticmethod
    def parse_transformer_model() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Transformer models argument parser.")
        parser.add_argument("-tr", "--train_file", default='./data/IK_NLP_22_PESTYLE/train.tsv',
                            type=str, help="Location of training file.")
        parser.add_argument("-te", "--test_file", default='./data/IK_NLP_22_PESTYLE/test.tsv',
                            type=str, help="Location of test file.")
        parser.add_argument("-o", "--outfile", default='./transformer_predictions.txt',
                            type=str, help="Path to write output to.")
        parser.add_argument("-m", "--model",
                            type=str, help="Specify an already fine-tuned model")
        return parser.parse_args()


def mkdirs(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        print("Creation of the directory %s failed" % dir_path)


def get_files(texts, filename):
    for i, text in enumerate(texts):
        with open(filename % i, "w") as txtfile:
            txtfile.write(text)


def load_dataframe(split: Split = None, mode: ConfigMode = ConfigMode.FULL,
                   exclude_modality: Modality = Modality.SCRATCH, only_numeric=False):
    """Load and filter the dataset."""
    df = load_dataset("GroNLP/ik-nlp-22_pestyle", mode, data_dir="IK_NLP_22_PESTYLE")[split] \
        .to_pandas().drop("item_id", axis=1, inplace=True)

    # filter modalities
    df_mod_filtered = df[df["modality"] != exclude_modality]

    # filter datatypes
    if only_numeric:
        numeric_type = ["float32", "int32"]
        df_mod_filtered = df_mod_filtered[
            [col for col in df_mod_filtered.columns if df_mod_filtered[col].dtype in numeric_type]].fillna(0)

    return df_mod_filtered


def get_train_labels():
    df = load_dataset("GroNLP/ik-nlp-22_pestyle", ConfigMode.MASK_SUBJECT, data_dir="IK_NLP_22_PESTYLE")[Split.TRAIN] \
        .to_pandas()
    y = np.array(df.subject_id)
    label_encoder = LabelBinarizer().fit(y)

    return y, label_encoder


def scaling(X_train, X_valid):
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_valid = scale.transform(X_valid)
    return X_train, X_valid


def data_split(X, y=None, test_size=0.1):
    if y == None:
        y = np.zeros(len(X))
    return train_test_split(X, y, test_size=test_size, random_state=42)


def get_tf_dataset(X, Y):
    def gen():
        for x, y in zip(X, Y):
            yield x, y

    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.int64),
        (tf.TensorShape([None]), tf.TensorShape([None])),
    )
    return dataset


def get_Kfolds(X, y, K=10):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    return skf.split(X, y)


def get_ling_feats():
    df_mt = pd.read_csv("../data/linguistic_features/train_mt.csv", sep="\t", index_col="Filename")
    df_tgt = pd.read_csv("../data/linguistic_features/train_tgt.csv", sep="\t", index_col="Filename")
    df_mt_test = pd.read_csv("../data/linguistic_features/test_mt.csv", sep="\t", index_col="Filename")
    df_tgt_test = pd.read_csv("../data/linguistic_features/test_tgt.csv", sep="\t", index_col="Filename")
    columns = (
            set(list(df_mt.columns))
            & set(list(df_tgt.columns))
            & set(list(df_mt_test.columns))
            & set(list(df_tgt_test.columns))
    )
    df_mt = df_mt[columns]
    df_mt_test = df_mt_test[columns]
    df_tgt = df_tgt[columns]
    df_tgt_test = df_tgt_test[columns]
    return df_mt.subtract(df_tgt, axis="columns"), df_mt_test.subtract(df_tgt_test, axis="columns")


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


def filter_features(df, th=0.9, verbose=True):
    N = len(df)
    df2 = pd.DataFrame()
    for column in df.columns:
        count = (df[column] == 0).sum()
        if count / N < (1 - th):
            df2[column] = df[column]
    if verbose:
        print(df2)
        print(len(df.columns), len(df2.columns))
        print(df2.columns)
    return df2


def output_test_predictions(model: Union[LinearRegression, RandomForestClassifier],
                            predictions: Union[np.array, list], experiment_type: str) -> None:
    """Generate txt file that contains all the predictions in a sequence within a single row."""
    with open(f"predictions/{model._estimator_type}_{experiment_type}_predictions.txt", 'w') as f:
        f.write(predictions)


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
