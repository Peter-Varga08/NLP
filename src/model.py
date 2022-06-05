import argparse
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import BorderlineSMOTE
from matplotlib import pyplot as plt
from seqeval.metrics import classification_report
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

import utils_pipeline


class NeuralNetwork:
    def __init__(self, args: argparse.Namespace, layers=[512, 256]):
        self.args = args
        self.layers = layers

        self.input_layer = Input(shape=(self.input_len,))
        self.dense = Dense(self.layers[0], activation="relu")(self.input_layer)
        self.dense = BatchNormalization()(self.dense)
        self.dense = Dropout(0.1)(self.dense)
        for l in self.layers[1:]:
            self.dense = Dense(l, activation="relu")(self.dense)
            self.dense = BatchNormalization()(self.dense)
            self.dense = Dropout(0.1)(self.dense)
        self.output_layer = Dense(3, activation="softmax")(self.dense)
        self.model = self.model()
        self.summary = self.summary()
        self.compile_model()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self):
        self._model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self):
        self._summary = self.model.summary()

    def compile_model(self) -> None:
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.args.lr, clipvalue=self.args.clipvalue),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    @classmethod
    def get_tf_dataset(cls):
        pass

    def get_early_stopping(self):
        return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                min_delta=0,
                                                patience=self.args.patience,
                                                verbose=1,
                                                mode='max',
                                                baseline=None,
                                                restore_best_weights=True)

    def fit(self, X_train, y_train) -> None:
        X_train = NeuralNetwork.get_tf_dataset(X_train, y_train).shuffle(500).batch(self.args.batch_size)
        self.model.fit(train_set,
                       validation_data=valid_set,
                       epochs=1000,
                       callbacks=[self.get_early_stopping()])
        return ...

    def predict(self, X_valid) -> None:
        X_valid = utils_pipeline.get_tf_dataset(X_valid, y_valid).batch(args.batch_size)
        return self.model.predict(X_valid)

    def evaluate(self, X_valid, y_valid):
        X_valid = utils_pipeline.get_tf_dataset(X_valid, y_valid).batch(args.batch_size)
        self.model.evaluate(X_valid, y_valid)

    # TODO
    def do_kfold_scoring(self, X: pd.DataFrame = None, y: np.ndarray = None, label_encoder=label_encoder,
                         input_len=input_len, layers=[512, 256], LR=1e-3, K=10, batch_size=16, oversampling=False):
        """Performs a k-fold CV with given model on the supplied dataset"""
        X = X.to_numpy()
        kfolds = utils_pipeline.get_Kfolds(X, y, K)
        scores_valid = []
        i = 0
        early_stop = NeuralNetwork.get_early_stopping()
        for train_index, valid_index in kfolds:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = label_encoder.transform(y[train_index]), label_encoder.transform(y[valid_index])
            X_train, X_valid = utils_pipeline.scaling(X_train, X_valid)

            model = NeuralNetwork(input_len, layers=layers, LR=LR)
            history = model.fit(train_set,
                                validation_data=valid_set,
                                epochs=1000,
                                callbacks=[early_stop])
            plt.figure(figsize=(20, 20))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.savefig(result_path + '/loss_%s.png' % i)
            plt.figure(figsize=(20, 20))
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.savefig(result_path + '/acc_%s.png' % i)

            preds = model.predict(X_valid)
            print(
                classification_report([np.argmax(gt) for gt in y_valid],
                                      [np.argmax(yp) for yp in preds]))
            abc = model.evaluate(X_valid, y_valid)


    # TODO
    def do_single(self, X, y, label_encoder, input_len, layers=[512, 256], LR=1e-3, batch_size=16, oversampling=False,
                  result_path='./w/best', patience=100):
        X = X.to_numpy()
        X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                              y,
                                                              test_size=0.1,
                                                              random_state=42)
        y_train, y_valid = label_encoder.transform(
            y_train), label_encoder.transform(y_valid)
        X_train, X_valid = utils_pipeline.scaling(X_train, X_valid)
        if oversampling:
            bsmote = BorderlineSMOTE(random_state=101, kind='borderline-1')
            X_train, y_train = bsmote.fit_resample(X_train, y_train)
        train_set = utils_pipeline.get_tf_dataset(X_train, y_train).shuffle(500).batch(batch_size)
        valid_set = utils_pipeline.get_tf_dataset(X_valid, y_valid).batch(batch_size)
        model = NeuralNetwork(input_len, layers=layers, LR=LR)
        early_stop = NeuralNetwork.get_early_stopping(patience=patience)
        history = model.fit(train_set,
                            validation_data=valid_set,
                            epochs=1000,
                            callbacks=[early_stop])
        model.save(result_path + '/best')
        plt.figure(figsize=(20, 20))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig(result_path + '/loss.png')
        plt.figure(figsize=(20, 20))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.savefig(result_path + '/acc.png')
        preds = model.predict(X_valid)
        print(classification_report([np.argmax(gt) for gt in y_valid],
                                    [np.argmax(yp) for yp in preds]))
        abc = model.evaluate(X_valid, y_valid)
        print("Results in terms of val_loss and val_accuracy:", abc)
        return abc


# TODO
class ClassificationTransformer(ClassificationModel):
    def __init__(self, base_args: argparse.Namespace, model_args: ClassificationArgs) -> None:
        self.base_args = base_args
        self.model_args = model_args
        super().__init__(self, "bert", self.base_args.bert_config, self.model_args, use_cuda=False, num_labels=3)

    @staticmethod
    def get_accuracy(gold: str, pred):
        """
        Retrieve the number of correctly predicted post-editors
        """
        # Map labels to ints, to streamline with model predictions
        gold_ints = []
        for g in gold:
            if g == "t1":
                gold_ints.append(0)
            elif g == "t2":
                gold_ints.append(1)
            elif g == "t3":
                gold_ints.append(2)
            continue

        # Get number of correct predictions
        correct = 0
        if len(gold_ints) == len(pred):
            for g, p in zip(gold_ints, pred):
                if g == p:
                    correct += 1
        else:
            print("ERROR: Gold standard and predictions are not of equal length!")

        return correct / len(gold)

    @staticmethod
    def _reformat_test(pairs: np.ndarray) -> pd.DataFrame:
        """
        Retrieve a regular list from the numpy array
        The output is a list that contains lists ([MT,PE]) with each sentence pair.
        """
        data = []
        mt = [t[0] for t in pairs]
        pe = [t[1] for t in pairs]

        for m, p in zip(mt, pe):
            data.append([m, p])

        test_df = pd.DataFrame(data)
        test_df.columns = ["text_mt", "text_pe"]
        return test_df

    @staticmethod
    def _reformat_train(pairs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """
        Retrieve a regular list from the numpy array
        The output is a list that contains lists ([MT,PE,label]) from each sentence pair.
        """
        data = []
        mt: List[str] = [t[0] for t in pairs]
        pe: List[str] = [t[1] for t in pairs]
        labs: List[int] = labels.flatten().tolist()

        for m, p, l in zip(mt, pe, labs):
            data.append([m, p, l])

        train_df = pd.DataFrame(data)
        train_df.columns = ["text_mt", "text_pe", "labels"]
        return train_df

    def fit(self, X_train, y_train):
        """
        Fit 'simpletransformer' classification model.
        """
        train_df = self._reformat_train(X_train, y_train)
        self.train_model(train_df)
