import os
from typing import Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from enums import ConfigMode, Modality, Split


def mkdirs(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        print("Creation of the directory %s failed" % dir_path)


def load_dataframe(split: Split = None, mode: ConfigMode = ConfigMode.FULL,
                   exclude_modality: Modality = Modality.SCRATCH, only_numeric=False):
    """Load and filter the dataset."""
    data = load_dataset("GroNLP/ik-nlp-22_pestyle", mode, data_dir="IK_NLP_22_PESTYLE")[split]
    df = data.to_pandas().drop("item_id", axis=1, inplace=True)

    # filter modalities
    df_mod_filtered = df[df["modality"] != exclude_modality]

    # filter datatypes
    if only_numeric:
        numeric_type = ["float32", "int32"]
        df_mod_filtered = df_mod_filtered[
            [col for col in df_mod_filtered.columns if df_mod_filtered[col].dtype in numeric_type]].fillna(0)

    return df_mod_filtered


def get_train_labels():
    data = load_dataset("GroNLP/ik-nlp-22_pestyle", ConfigMode.MASK_SUBJECT, data_dir="IK_NLP_22_PESTYLE")[Split.TRAIN]
    df = data.to_pandas()
    return LabelBinarizer().fit_transform(np.array(df.subject_id))


def get_train_label_encoder():
    data = load_dataset("GroNLP/ik-nlp-22_pestyle", ConfigMode.MASK_SUBJECT, data_dir="IK_NLP_22_PESTYLE")[Split.TRAIN]
    df = data.to_pandas()
    encoder = LabelBinarizer()
    encoder.fit(np.array(df.subject_id))
    return encoder


# TODO: Handle scaling for test set too
def scaling(X_train, X_valid):
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_valid = scale.transform(X_valid)
    return X_train, X_valid


def get_ling_feats(split: Split = None):
    if not isinstance(split, Split):
        raise TypeError(f"'split' expected type 'Split', got {type(split)}.")

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

    if split == Split.TRAIN:
        df_mt = df_mt[columns]
        df_tgt = df_tgt[columns]
        return df_mt.subtract(df_tgt, axis="columns")

    df_mt_test = df_mt_test[columns]
    df_tgt_test = df_tgt_test[columns]
    return df_mt_test.subtract(df_tgt_test, axis="columns")


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


def run_test(model, X_test, dataset_type: str) -> None:
    """Run model on test set and obtain predictions."""
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


def get_features_sorted(selector: SelectKBest) -> List[Tuple[str, float]]:
    """Return sorted features and their relevance according to a KBest selector."""
    return sorted(zip(list(selector.get_feature_names_out()), list(selector.scores_)),
                  key=lambda x: x[1], reverse=True)
