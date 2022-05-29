"""This script is to evaluate feature importance of the neural network with Integrated Gradients :) """

import utils
import numpy as np
import tensorflow as tf
import data_management
import pandas as pd

if __name__ == "__main__":

    output_path, features, *_ = utils.parse()
    model = tf.keras.models.load_model(output_path + '/best')

    if features == 'ling':
        df_train, df_test = data_management.get_ling_feats()
    else:
        df_train, *_ = data_management.load_dataframe(_set='train', mode='full', exclude_modality='ht', only_numeric=True)
        df_test, *_ = data_management.load_dataframe(_set='test', mode='mask_subject', exclude_modality='ht',
                                                     only_numeric=True,
                                                     verbose=True)
        if features == 'both':
            df_train_ling, df_test_ling = data_management.get_ling_feats()
            df_train_ling.index, df_test_ling.index = df_train.index, df_test.index
            df_train = pd.concat([df_train, df_train_ling], axis='columns')
            df_test = pd.concat([df_test, df_test_ling], axis='columns')

    X_train, X_test = data_management.scaling(df_train, df_test)
    predictions = model.predict(X_test).argmax(axis=1)

    with open(output_path + '/predictions.txt', 'w') as txt:
        for pred in predictions:
            txt.write(pred)
