
from utils import *
import numpy as np
import tensorflow as tf
from data_management import *

print('Hello World! This script is to evaluate feature importance of the neural network with Integrated Gradients :)')
result_path, features, _, _ = parse()
model = tf.keras.models.load_model(result_path+'/best') 

if features == 'ling':
    df_train, df_test = get_ling_feats()
else:
    df_train, _, _ = load_dataframe(_set='train', mode='full', exclude_modality='ht', only_numeric=True)
    df_test, _, _ = load_dataframe(_set='test', mode='mask_subject', exclude_modality='ht', only_numeric=True, verbose=True)
    if features == 'both':
        df_train2, df_test2 = get_ling_feats() 
        df_train2.index = df_train.index
        df_train = pd.concat([df_train, df_train2], axis='columns')
        df_test2.index = df_test.index
        df_test = pd.concat([df_test, df_test2], axis='columns')

columns = list(df_test.columns)
input_len = len(columns)
print(columns)

X_train, X_test = scaling(df_train, df_test)
predictions = model.predict(X_test).argmax(axis=1) 

with open(result_path + '/predictions.txt', 'w') as txt:
    for pred in predictions:
        txt.write(pred)
