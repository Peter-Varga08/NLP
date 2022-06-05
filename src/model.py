import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input


#TODO: Finish
class RegressionModel:
    if args.model == 'lr':
        model = LinearRegression(normalize=args.normalize,
                                 n_jobs=args.n_jobs)
    elif args.model == 'rr':
        model = Ridge(normalize=args.normalize,
                      alpha=args.alpha,
                      solver=args.solver)
    else:
        model = ElasticNet(normalize=args.normalize,
                           l1_ratio=args.l1_ratio,
                           precompute=True,
                           max_iter=args.max_iter,
                           tol=args.tol,
                           selection=args.selection)

class NeuralNetwork:
    def __init__(self, input_len: int = None, lr: float = None, clipvalue: float = None, layers=(512, 256)):
        self.lr = lr
        self.clipvalue = clipvalue
        self.input_len = input_len
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
        self.train_history: Dict[str, np.ndarray] = {}

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, clipvalue=self.clipvalue),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    @staticmethod
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

    def get_early_stopping(self):
        return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                min_delta=0,
                                                patience=self.args.patience,
                                                verbose=1,
                                                mode='max',
                                                baseline=None,
                                                restore_best_weights=True)

    # TODO: How to use X_valid and y_valid without passing them
    def fit(self, X_train, y_train, X_valid, y_valid) -> None:
        X_y_train = NeuralNetwork.get_tf_dataset(X_train, y_train).shuffle(500).batch(self.args.batch_size)

        X_y_valid = NeuralNetwork.get_tf_dataset(X_valid, y_valid).shuffle(500).batch(self.args.batch_size)
        self.train_history = self.model.fit(X_y_train,
                                            validation_data=X_y_valid,
                                            epochs=1000,
                                            callbacks=[self.get_early_stopping()]).history

    def predict(self, X_valid: np.ndarray) -> np.ndarray:
        return self.model.predict(X_valid)

    def evaluate(self, X_valid: np.ndarray, y_valid: np.ndarray) -> Any:
        return self.model.evaluate(X_valid, y_valid)


# TODO
class ClassificationTransformer(ClassificationModel):
    def __init__(self, bert_config: str, model_args: ClassificationArgs) -> None:
        self.bert_config = bert_config
        self.model_args = model_args
        super().__init__(self, "bert", self.bert_config, self.model_args, use_cuda=False, num_labels=3)

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
