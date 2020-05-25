import numpy as np
import tensorflow as tf
import time
import random

def generate_data(size):
    X = tf.random.uniform([size, 2], -2, maxval=2, dtype=tf.dtypes.int32)
    Y = []
    for xi in X:
        if xi[0] * xi[0] > xi[1]:
            Y.append(1)
        else:
            Y.append(0)
    return np.array(X), np.array(Y)
    
if __name__ == '__main__':
    # tf.random.uniform([1000, 2], -2, maxval=2, dtype=tf.dtypes.int32)
    train_x, train_y = generate_data(1000)
    test_x, test_y = generate_data(100)

    # building layers
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.5),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.binary_accuracy]
    )

    train_start = time.time()
    result = model.fit(train_x, train_y, batch_size=1000, epochs=1000, verbose=0)
    train_loss = result.history['loss'][-1]
    train_accuracy = result.history['binary_accuracy'][-1]
    train_end = time.time()

    test_start = time.time()
    result = model.evaluate(test_x, test_y, verbose=0)
    test_loss = result[0]
    test_accuracy = result[1]
    test_end = time.time()

    print('Train time: ' + str(train_end - train_start) + 's')
    print("Train loss: {}".format(train_loss))
    print("Train accuracy: {}".format(train_accuracy))

    print('Test time: ' + str(test_end - test_start) + 's')
    print("Test loss: {}".format(test_loss))
    print("Test accuracy: {}".format(test_accuracy))