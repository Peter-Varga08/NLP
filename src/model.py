import gc
from typing import Any, Dict, List

import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input


class NeuralNetwork:
    def __init__(
        self,
        input_len: int = None,
        output_len: int = None,
        patience: int = 100,
        lr: float = 1e-3,
        clipvalue: float = 0.5,
        layers=(512, 256),
        batch_size: int = 16,
        dropout: float = 0.1,
        epochs: int = 1000,
        optimizer: str = "Adam",
    ):
        self.input_len = input_len
        self.output_len = output_len
        self.lr = lr
        self.clipvalue = clipvalue
        self.layers = layers
        self.patience = patience
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizers = {
            "Adam": tf.keras.optimizers.Adam(
                learning_rate=self.lr, clipvalue=self.clipvalue
            ),
            "RMSprop": tf.keras.optimizers.RMSprop(
                learning_rate=self.lr, clipvalue=self.clipvalue
            ),
        }
        self.optimizer = self.optimizers[optimizer]
        self.dropout = dropout

        self.input_layer = Input(shape=(self.input_len,))
        self.dense = Dense(self.layers[0], activation="relu")(self.input_layer)
        self.dense = BatchNormalization()(self.dense)
        self.dense = Dropout(self.dropout)(self.dense)
        for layer in self.layers[1:]:
            self.dense = Dense(layer, activation="relu")(self.dense)
            self.dense = BatchNormalization()(self.dense)
            self.dense = Dropout(self.dropout)(self.dense)
        self.output_layer = Dense(self.output_len, activation="softmax")(self.dense)

        print("Compiling keras model...")
        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)
        self.compile_model()
        print("Model build has completed.")
        self.train_history: Dict[str, NDArray] = {}

    def __del__(self) -> None:
        tf.keras.backend.clear_session()
        gc.collect()
        if hasattr(self, "model"):
            del self.model

    def summary(self) -> Any:
        return self.model.summary()

    def compile_model(self) -> None:
        self.model.compile(
            optimizer=self.optimizer,
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
            epochs=self.epochs,
            callbacks=[self.get_early_stopping()],
        ).history

    def predict(self, X: NDArray) -> NDArray:
        return self.model.predict(X)

    def evaluate(self, X: NDArray, y: NDArray) -> Any:
        return self.model.evaluate(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


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


def get_model_classname(model: Any) -> str:
    """Utility function to return the name of the class of the model."""
    return model.__class__.__name__
