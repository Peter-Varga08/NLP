import os
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from enums import ConfigMode, Split

DATA_PATH = "../data/"


def mkdirs(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        print("Creation of the directory %s failed" % dir_path)


def check_and_get_split(func: Callable):
    def wrapper_function(*args):
        if not isinstance(args[0], Split):
            raise TypeError(
                f"Positional argument {args[0]} has to be of type {type(Split)}."
            )
        elif len(args) != 1:
            raise TypeError(
                f"Function takes {len(args)} positional argument, but 1 was given."
            )
        else:
            split = args[0]
            if split == Split.TRAIN:
                return func("train", ConfigMode.FULL.value)
            elif split == Split.TEST:
                return func("test", ConfigMode.MASK_MODALITY.value)
            else:
                raise ValueError("Wrong input value for argument split.")

    return wrapper_function


@check_and_get_split
def split_load_dataset(split: Union[Split, str], mode: ConfigMode):
    data = load_dataset("GroNLP/ik-nlp-22_pestyle", mode, data_dir=DATA_PATH)[split]
    df = data.to_pandas().drop("item_id", axis=1)
    return df[df["modality"] != "ht"].reset_index(drop=True)


def load_dataframe(split: Split = None, only_numeric=False):
    """Load and filter the dataset."""
    df = split_load_dataset(split)
    # filter datatypes
    if only_numeric:
        numeric_type = ["float32", "int32"]
        df = df[[col for col in df.columns if df[col].dtype in numeric_type]].fillna(0)
    return df


def get_labels(split: Split):
    return np.array(split_load_dataset(split).subject_id)


def get_label_encoder(split: Split):
    df = split_load_dataset(split)
    encoder = LabelBinarizer()
    encoder.fit(np.array(df.subject_id))
    return encoder


# TODO: Handle scaling for test set too
def scaling(X_train, *X_valid):
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    if len(X_valid) == 1 and isinstance(X_valid[0], ndarray):
        X_valid = X_valid[0]
        X_valid = scale.transform(X_valid)
        return X_train, X_valid
    return X_train


def get_ling_feats(split: Split = None):
    if not isinstance(split, Split):
        raise TypeError(f"'split' expected type 'Split', got {type(split)}.")

    df_mt = pd.read_csv(
        "../../data/linguistic_features/train_mt.csv", sep="\t", index_col="Filename"
    )
    df_tgt = pd.read_csv(
        "../../data/linguistic_features/train_tgt.csv", sep="\t", index_col="Filename"
    )
    df_mt_test = pd.read_csv(
        "../../data/linguistic_features/test_mt.csv", sep="\t", index_col="Filename"
    )
    df_tgt_test = pd.read_csv(
        "../../data/linguistic_features/test_tgt.csv", sep="\t", index_col="Filename"
    )
    columns = (
        set(list(df_mt.columns))
        & set(list(df_tgt.columns))
        & set(list(df_mt_test.columns))
        & set(list(df_tgt_test.columns))
    )

    if split == Split.TRAIN:
        df_mt = df_mt[columns]
        df_tgt = df_tgt[columns]
        return df_mt.subtract(df_tgt, axis="columns")

    df_mt_test = df_mt_test[columns]
    df_tgt_test = df_tgt_test[columns]
    return df_mt_test.subtract(df_tgt_test, axis="columns")


def output_test_predictions(
    model: Union[LinearRegression, RandomForestClassifier],
    predictions: Union[np.array, list],
    experiment_type: str,
) -> None:
    """Generate txt file that contains all the predictions in a sequence within a single row."""
    with open(
        f"predictions/{model._estimator_type}_{experiment_type}_predictions.txt", "w"
    ) as f:
        f.write(predictions)


def save_feature_importance_plots(
    forest_importances: pd.Series, std: float, experiment_type: str
) -> None:
    """Plot and save feature importances of RandomForest model."""
    fig, ax = plt.subplots()
    # fig.tight_layout(h_pad=10)
    forest_importances.plot.barh(yerr=std, ax=ax)
    ax.set_title("MDI-based Feature Importances")
    ax.set_ylabel("Features")
    ax.set_xlabel("Mean decrease in impurity")
    fig.savefig(f"../images/RandomForest_{experiment_type}_importances.jpg")


def transform_predictions(predictions: np.ndarray) -> str:
    """Transform one-hot encoded labels into integer labels."""
    transformed_predictions = []
    for vector in predictions:
        transformed_predictions.append(str(np.argmax(vector)))
    return "".join(transformed_predictions)


def softmax2onehot(predictions: np.ndarray) -> np.ndarray:
    """Convert softmax values within an array of vectors to one-hot encoding, based on argmax."""
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


def get_features_sorted(selector: SelectKBest) -> List[Tuple[str, float]]:
    """Return sorted features and their relevance according to a KBest selector."""
    return sorted(
        zip(list(selector.get_feature_names_out()), list(selector.scores_)),
        key=lambda x: x[1],
        reverse=True,
    )
