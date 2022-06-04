import argparse

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import BorderlineSMOTE
from matplotlib import pyplot as plt
from seqeval.metrics import classification_report
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

import utils_pipeline


class NeuralNetwork:
    def __init__(self, input_len, layers, lr):
        self.lr = lr
        self.input_layer = Input(shape=(input_len,))
        self.dense = Dense(layers[0], activation="relu")(self.input_layer)
        self.dense = BatchNormalization()(self.dense)
        self.dense = Dropout(0.1)(self.dense)
        for l in layers[1:]:
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, clipvalue=0.5),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    @classmethod
    def get_early_stopping(cls, patience=100) -> None:
        return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                min_delta=0,
                                                patience=patience,
                                                verbose=1,
                                                mode='max',
                                                baseline=None,
                                                restore_best_weights=True)

    #TODO
    def do_kfold_scoring(self, X: pd.DataFrame = None,
                         y: np.ndarray = None,
                         label_encoder=label_encoder,
                         input_len=input_len,
                         layers=[512, 256],
                         LR=1e-3,
                         K=10,
                         batch_size=16,
                         oversampling=False):
        """Performs a k-fold CV with given model on the supplied dataset"""
        X = X.to_numpy()
        kfolds = utils_pipeline.get_Kfolds(X, y, K)
        scores_valid = []
        i = 0
        early_stop = NeuralNetwork.get_early_stopping()
        for train_index, valid_index in kfolds:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = label_encoder.transform(
                y[train_index]), label_encoder.transform(y[valid_index])
            X_train, X_valid = utils_pipeline.scaling(X_train, X_valid)
            if oversampling:
                bsmote = BorderlineSMOTE(random_state=101, kind='borderline-1')
                X_train, y_train = bsmote.fit_resample(X_train, y_train)
            train_set = utils_pipeline.get_tf_dataset(X_train,
                                                      y_train).shuffle(500).batch(batch_size)
            valid_set = utils_pipeline.get_tf_dataset(X_valid, y_valid).batch(batch_size)
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
            print(abc)
            scores_valid.append(abc)
            i += 1
        losses = [el[0] for el in scores_valid]
        accs = [el[1] for el in scores_valid]
        print("Average loss score:", round(np.mean(losses), 4), '+-',
              round(np.std(losses), 4))
        print("Average acc score:", round(np.mean(accs), 4), '+-',
              round(np.std(accs), 4))

    #TODO
    def do_single(self, X,
                  y,
                  label_encoder,
                  input_len,
                  layers=[512, 256],
                  LR=1e-3,
                  batch_size=16,
                  oversampling=False,
                  result_path='./w/best',
                  patience=100):
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


#TODO
class Transformer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = ClassificationModel("bert", self.args.model, use_cuda=False)

    @staticmethod
    def arg_parser():


    @staticmethod
    def prepare_train(train_file: str, include_from_scratch=False):
        """
        Extract the relevant features from the data set: MT, PE and subject.
        If include_from_scratch is False, we do not take into account the HT modality.
        """
        pairs = []
        labels = []
        with open(train_file) as train:
            train.readline()  # Skip first line: this is the header
            for line in train:
                if line.split("\t")[1][0] == "t":
                    # Get a list of: [MT output, post-edited sentence, subject id]
                    if include_from_scratch:
                        pairs.append([line.split("\t")[4], line.split("\t")[5]])
                        labels.append(line.split("\t")[1])
                    else:
                        if line.split("\t")[4]:
                            pairs.append([line.split("\t")[4], line.split("\t")[5]])
                            labels.append(line.split("\t")[1])

        # We use only 90% of the data, in order to allow comparison with the other models
        X_train, _, y_train, _ = train_test_split(pairs, labels, test_size=0.1, random_state=42)

        return X_train, y_train

    @staticmethod
    def predict_test(test_data: str, model: ClassificationModel, include_from_scratch=False, final=False):
        """
        Given a fine-tuned model, predict the labels (translators) from MT-PE sentence pairs.
        If include_from_scratch is False, we do not take into account the HT modality.
        """
        pairs = []
        predictions = []

        if final:  # Use the 'final' (official) test set
            with open(test_data) as test:
                test.readline()  # Skip first line: this is the header
                for line in test:
                    if line.split("\t")[0][0] in "0123456789":
                        if include_from_scratch:
                            pairs.append([line.split("\t")[4], line.split("\t")[5]])
                        else:
                            if line.split("\t")[4]:
                                pairs.append([line.split("\t")[4], line.split("\t")[5]])
        else:  # Use a subset of the data for cross-validation
            pairs = test_data

        for pair in pairs:
            prediction, _ = model.predict(pair)
            predictions.append(prediction[0])

        return predictions

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
    def reformat_test(pairs):
        """
        Retrieve a regular list from the numpy array
        The output is a list that contains lists ([MT,PE]) with each sentence pair.
        """
        data = []
        mt = [t[0] for t in pairs]
        pe = [t[1] for t in pairs]

        for m, p in zip(mt, pe):
            data.append([m, p])

        return data

    @staticmethod
    def reformat_train(pairs, labels):
        """
        Retrieve a regular list from the numpy array
        The output is a list that contains lists ([MT,PE,label]) from each sentence pair.
        """
        data = []
        mt = [t[0] for t in pairs]
        pe = [t[1] for t in pairs]
        labs = np.ndarray.tolist(labels)

        for m, p, l in zip(mt, pe, labs):
            data.append([m, p, l])

        return data
