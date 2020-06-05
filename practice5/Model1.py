# Model1.py

import tensorflow as tf
from tensorflow import keras
import numpy as np


if __name__ == '__main__':
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    x_train, x_test = x_train/255.0, x_test/255.0   # scaling; pre-processing
    y_train, y_test = x_train, x_test   # ground-truth data

    model = tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(0.1, input_shape=(28, 28)),    # add gaussian noise here
    ])
    input_shape1 = (3, 3, 3, 64)
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape1))
    input_shape2 = (64, 3, 3, 64)
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape2))
    input_shape3 = (64, 3, 3, 64)
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape3))
    input_shape4 = (64, 3, 3, 64)
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape4))
    input_shape5 = (64, 3, 3, 3)
    model.add(tf.keras.layers.Conv2D(3, 3, padding='same', activation=None, input_shape=input_shape5))

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam,
        loss=tf.keras.losses.MeanSquaredError()
    )
    model.fit(x_train, y_train, batch_size=32, epochs=100)