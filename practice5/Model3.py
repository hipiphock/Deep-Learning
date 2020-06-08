# Model3.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib
import numpy as np
from google.colab import files, drive

def construct_model():
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Lambda(lambda input_tensor: input_tensor)(input_tensor)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(3, 3, padding='same')(x)
    model = tf.keras.Model(input_tensor, x)
    return model

if __name__ == '__main__':

    # image test
    drive.mount('/DeepLearning/', force_remount=True)
    root_dir = "/DeepLearning/"
    img = image.load_img('noisy.png')
    img = image.img_to_array(img)

    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0   # scaling; pre-processing
    y_train = x_train.copy()
    y_test = x_test.copy()
    x_train += np.random.normal(0, .1, x_train.shape)

    model = construct_model()

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError()
    )
    model.fit(x_train, y_train, batch_size=32, epochs=100)

    result = model.evaluate(x_test, y_test, verbose=0)
    model.save_weights('model3')
    print(result)

    for i in range(0, img.shape[0], 32):
        for j in range(0, img.shape[1], 32):
            img[i:i + 32, j:j + 32] = model.predict(np.expand_dims(img[i:i + 32, j:j + 32], 0))
    img = image.array_to_img(img)
    img.save('Model3.png')