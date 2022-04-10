import sys
from utils import *
import numpy as np
import tensorflow as tf
from data_management import *
import matplotlib.pyplot as plt
from modeling import *
from sklearn.metrics import classification_report
from imblearn.over_sampling import BorderlineSMOTE

def get_early_stopping(patience=100):
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=patience,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )

def do_kfold_scoring(X, y, label_encoder, input_len, layers=[512, 256], LR=1e-3, K=10, batch_size=16, oversampling=False):
    """Performs a k-fold CV with given model on the supplied dataset"""
    X = X.to_numpy()
    kfolds = get_Kfolds(X, y, K)
    scores_valid = []
    i = 0
    early_stop = get_early_stopping()
    for train_index, valid_index in kfolds:
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = label_encoder.transform(y[train_index]), label_encoder.transform(y[valid_index])
        X_train, X_valid = scaling(X_train, X_valid)
        if oversampling:
            bsmote = BorderlineSMOTE(random_state = 101, kind = 'borderline-1')
            X_train, y_train = bsmote.fit_resample(X_train, y_train)
        train_set = get_tf_dataset(X_train, y_train).shuffle(500).batch(batch_size)
        valid_set = get_tf_dataset(X_valid, y_valid).batch(batch_size)
        model = get_neural_network_model(input_len, layers=layers, LR=LR)
        history = model.fit(train_set, validation_data=valid_set, epochs=1000, callbacks=[early_stop])
        plt.figure(figsize=(20, 20))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig(result_path+'/loss_%s.png' % i)
        plt.figure(figsize=(20, 20))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.savefig(result_path+'/acc_%s.png' % i)

        preds = model.predict(X_valid)
        print(classification_report([np.argmax(gt) for gt in y_valid], [np.argmax(yp) for yp in preds])) 
        abc = model.evaluate(X_valid, y_valid)      
        print(abc)   
        scores_valid.append(abc)  
        i += 1
    losses = [el[0] for el in scores_valid]
    accs = [el[1] for el in scores_valid]
    print("Average loss score:", round(np.mean(losses), 4), '+-', round(np.std(losses), 4))
    print("Average acc score:", round(np.mean(accs), 4), '+-', round(np.std(accs), 4))

def do_single(X, y, label_encoder, input_len, layers=[512, 256], LR=1e-3, batch_size=16, oversampling=False, result_path='./w/best', patience=100):
    X = X.to_numpy()
    X_train, X_valid, y_train, y_valid= train_test_split(X, y, test_size=0.1, random_state=42)
    y_train, y_valid = label_encoder.transform(y_train), label_encoder.transform(y_valid)
    X_train, X_valid = scaling(X_train, X_valid)
    if oversampling:
        bsmote = BorderlineSMOTE(random_state = 101, kind = 'borderline-1')
        X_train, y_train = bsmote.fit_resample(X_train, y_train)
    train_set = get_tf_dataset(X_train, y_train).shuffle(500).batch(batch_size)
    valid_set = get_tf_dataset(X_valid, y_valid).batch(batch_size)
    model = get_neural_network_model(input_len, layers=layers, LR=LR)
    early_stop = get_early_stopping(patience)
    history = model.fit(train_set, validation_data=valid_set, epochs=1000, callbacks=[early_stop])
    model.save(result_path+'/best')
    plt.figure(figsize=(20, 20))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig(result_path+'/loss.png')
    plt.figure(figsize=(20, 20))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.savefig(result_path+'/acc.png')
    preds = model.predict(X_valid)
    print(classification_report([np.argmax(gt) for gt in y_valid], [np.argmax(yp) for yp in preds])) 
    abc = model.evaluate(X_valid, y_valid)   
    print("Results in terms of val_loss and val_accuracy:", abc)
    return abc

if __name__ == "__main__":
    print('Hello World! This script is to train the neural network :)')
    result_path, features, oversampling, kfold = parse()
    df_train, y, label_encoder = load_dataframe(_set='train', mode='full', exclude_modality='ht', only_numeric=True)
    print(features)
    if features == 'ling':
        df_train, _ = get_ling_feats()
        if kfold == 'L': 
            ths = [0, .5, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
            losses = []
            accs = []
            for th in ths:
                print(th)
                df2 = filter_features(df_train, th=th, verbose=False)
                columns = list(df2.columns)
                input_len = len(columns)
                print(input_len)
                if input_len == 0:
                    break
                abc = do_single(df2, y, label_encoder, input_len, patience=100, LR=1e-3, oversampling=oversampling, result_path=result_path)
                losses.append(abc[0])
                accs.append(abc[1])
            plt.figure(figsize=(15, 15))
            plt.plot(losses)
            plt.savefig(result_path+'/ling_loss.png')
            plt.figure(figsize=(15, 15))
            plt.plot(accs)
            plt.savefig(result_path+'/ling_acc.png')
            df_train = filter_features(df_train, th=ths[np.argmax(accs)], verbose=False)
        else:
            print(kfold)
    elif features == 'both':
       df_train2, _ = get_ling_feats() 
       print(df_train2.shape)
       df_train2 = filter_features(df_train2, th=0, verbose=False)
       print(df_train2.shape)
       df_train2.index = df_train.index
       df_train = pd.concat([df_train, df_train2], axis='columns')
       print(df_train.shape)
       print(df_train)
       #set_trace()
    columns = list(df_train.columns)
    input_len = len(columns)
    print(input_len)
    if kfold:
        do_kfold_scoring(df_train, y, label_encoder, input_len, oversampling=oversampling)
    do_single(df_train, y, label_encoder, input_len, oversampling=oversampling, result_path=result_path)
      