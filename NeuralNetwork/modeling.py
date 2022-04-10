import tensorflow as tf
from tensorflow.keras.layers import *

def get_neural_network_model(input_len, layers, LR):
    input_layer = Input(shape=(input_len,))
    dense = Dense(layers[0], activation='relu')(input_layer)
    dense = BatchNormalization()(dense)
    dense = Dropout(.1)(dense)
    for l in layers[1:]:
        dense = Dense(l, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(.1)(dense)
    class_out = Dense(3, activation='softmax')(dense)
    model = tf.keras.Model(inputs=input_layer, outputs=class_out)
    print(model.summary())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipvalue=.5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

