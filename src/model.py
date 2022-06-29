from typing import Any, Dict, List

import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input


# TODO: Finish
class RegressionModel(LinearRegression, Ridge, ElasticNet):
    def __init__(
        self,
        name: str = None,
        normalize: bool = True,
        n_jobs=-1,
        alpha: float = 0.5,
        l1_ratio: float = 0.5,
        precompute=True,
        max_iter: int = 100,
        tol: float = None,
        selection: str = "random",
    ):
        self.name = name
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection

        if self.name == "lr":
            self.model = LinearRegression(normalize=self.normalize, n_jobs=self.n_jobs)
            super(LinearRegression, self).__init__()

        elif self.name == "rr":
            self.model = Ridge(normalize=self.normalize, alpha=self.alpha)
            super(Ridge, self).__init__()
        else:
            self.model = ElasticNet(
                normalize=self.normalize,
                l1_ratio=self.l1_ratio,
                precompute=True,
                max_iter=self.max_iter,
                tol=self.tol,
                selection=self.selection,
            )
            super(ElasticNet, self).__init__()

    def __str__(self) -> str:
        return self.model.__class__.__name__

    def __repr__(self) -> str:
        return repr(self.model)


# class RandomForestClassifier_(RandomForestClassifier):
#     def __init__(self, kwargs) -> None:
#         super(RandomForestClassifier_).__init__(**kwargs)
#
#     def __str__(self):
#         return self.__class__.__name__


class NeuralNetwork:
    def __init__(
        self,
        input_len: int = None,
        lr: float = None,
        clipvalue: float = None,
        layers=(512, 256),
        patience: int = None,
        batch_size: int = None,
    ):
        self.lr = lr
        self.clipvalue = clipvalue
        self.input_len = input_len
        self.layers = layers
        self.patience = patience
        self.batch_size = batch_size

        self.input_layer = Input(shape=(self.input_len,))
        self.dense = Dense(self.layers[0], activation="relu")(self.input_layer)
        self.dense = BatchNormalization()(self.dense)
        self.dense = Dropout(0.1)(self.dense)
        for layer in self.layers[1:]:
            self.dense = Dense(layer, activation="relu")(self.dense)
            self.dense = BatchNormalization()(self.dense)
            self.dense = Dropout(0.1)(self.dense)
        self.output_layer = Dense(3, activation="softmax")(self.dense)

        self.model = self.model()
        self.summary = self.summary()
        self.compile_model()
        self.train_history: Dict[str, NDArray] = {}

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
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.lr, clipvalue=self.clipvalue
            ),
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
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,
            patience=self.patience,
            verbose=1,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )

    # TODO: How to use X_valid and y_valid without passing them
    def fit(self, X_train, y_train, X_valid, y_valid) -> None:
        X_y_train = (
            NeuralNetwork.get_tf_dataset(X_train, y_train)
            .shuffle(500)
            .batch(self.batch_size)
        )

        X_y_valid = (
            NeuralNetwork.get_tf_dataset(X_valid, y_valid)
            .shuffle(500)
            .batch(self.batch_size)
        )
        self.train_history = self.model.fit(
            X_y_train,
            validation_data=X_y_valid,
            epochs=1000,
            callbacks=[self.get_early_stopping()],
        ).history

    def predict(self, X_valid: NDArray) -> NDArray:
        return self.model.predict(X_valid)

    def evaluate(self, X_valid: NDArray, y_valid: NDArray) -> Any:
        return self.model.evaluate(X_valid, y_valid)


# TODO
class ClassificationTransformer(ClassificationModel):
    def __init__(self, bert_config: str, model_args: ClassificationArgs) -> None:
        self.bert_config = bert_config
        self.model_args = model_args
        super().__init__(
            self,
            "bert",
            self.bert_config,
            self.model_args,
            use_cuda=False,
            num_labels=3,
        )

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
    def _reformat_test(pairs: NDArray) -> pd.DataFrame:
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
    def _reformat_train(pairs: NDArray, labels: NDArray) -> pd.DataFrame:
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
