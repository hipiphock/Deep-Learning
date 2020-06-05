# Model2.py

import tensorflow as tf
from tensorflow import keras
import numpy as np


if __name__ == '__main__':
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    x_train, x_test = x_train/255.0, x_test/255.0   # scaling; pre-processing
    y_train, y_test = x_train, x_test   # ground-truth data

    model = tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(0.1, input_shape=(32, 32, 3)),    # add gaussian noise here
    ])
    input_shape = (32, 32, 3)
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(3, (3,3), padding='same', activation=None))

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError()
    )
    model.fit(x_train, y_train, batch_size=32, epochs=100)

    _, acc = model.evaluate(x_test, y_test, verbose=0)