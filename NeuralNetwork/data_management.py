import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from pdb import set_trace

def load_dataframe(_set='train', mode='full', exclude_modality=None, only_numeric=False, verbose=False):
    df = load_dataset(
        "GroNLP/ik-nlp-22_pestyle", 
        mode, 
        data_dir="IK_NLP_22_PESTYLE"
    )[_set].to_pandas()
    if exclude_modality != None:
        df = df[df['modality'] != exclude_modality] # exclude_modality = 'ht'
    if mode != 'mask_subject':
        y = np.array(df.subject_id)
        label_encoder = LabelBinarizer().fit(y)
    else:
        y = None
        label_encoder = None
    if only_numeric:
        numeric_type = ['float32', 'int32']
        df = df[[col for col in df.columns if df[col].dtype in numeric_type]].fillna(0)
    df.drop('item_id', axis=1, inplace=True)
    if verbose:
        print(df)
    return df, y, label_encoder

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
            label = y#translator2label(y)
            yield (x, label)

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
    df_mt = pd.read_csv('Linguistic_features/train_mt.csv', sep='\t', index_col='Filename')
    df_tgt = pd.read_csv('Linguistic_features/train_tgt.csv', sep='\t', index_col='Filename')
    df_mt_test = pd.read_csv('Linguistic_features/test_mt.csv', sep='\t', index_col='Filename')
    df_tgt_test = pd.read_csv('Linguistic_features/test_tgt.csv', sep='\t', index_col='Filename')
    columns = set(list(df_mt.columns)) & set(list(df_tgt.columns)) & set(list(df_mt_test.columns)) & set(list(df_tgt_test.columns))
    df_mt = df_mt[columns]
    df_mt_test = df_mt_test[columns]
    df_tgt = df_tgt[columns]
    df_tgt_test = df_tgt_test[columns]
    return df_mt.subtract(df_tgt, axis='columns'), df_mt_test.subtract(df_tgt_test, axis='columns')

def filter_features(df, th=.9, verbose=True):
    N = len(df)
    df2 = pd.DataFrame()
    for column in df.columns:
        count = (df[column] == 0).sum()
        if count / N < (1-th):
            df2[column] = df[column]
    if verbose:
        print(df2)
        print(len(df.columns), len(df2.columns))
        print(df2.columns)
    return df2