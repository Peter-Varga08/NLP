import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout


class NeuralNetwork:

    def __init__(self, input_len, layers, lr):
        self.input_layer = Input(shape=(input_len,))
        self.dense = Dense(layers[0], activation='relu')(self.input_layer)
        self.dense = BatchNormalization()(self.dense)
        self.dense = Dropout(.1)(self.dense)
        for l in layers[1:]:
            self.dense = Dense(l, activation='relu')(self.dense)
            self.dense = BatchNormalization()(self.dense)
            self.dense = Dropout(.1)(self.dense)
        self.output_layer = Dense(3, activation='softmax')(self.dense)
        self._model = self.model()
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

    def compile_model(self):
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipvalue=.5),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

