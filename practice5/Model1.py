# Model1.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib
import numpy as np

## FLAGS
TRAINED = False

def construct_model():
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Lambda(lambda input_tensor: input_tensor)(input_tensor)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(3, 3, padding='same')(x)
    model = tf.keras.Model(input_tensor, x)
    return model

if __name__ == '__main__':
    # make model
    model = construct_model()
    model.summary()

    # train
    if TRAINED == False:
        (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0   # scaling; pre-processing
        y_train = x_train.copy()
        y_test = x_test.copy()
        x_train += np.random.normal(0, .1, x_train.shape)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError()
        )
        model.fit(x_train, y_train, batch_size=32, epochs=100)
        result = model.evaluate(x_test, y_test, verbose=0)
        print(result)

        model.save_weights('model1')
        TRAINED = True
    
    else:
        model.load_weights('model1')

    # evaluate

    # image
    img = image.load_img('noisy.png')
    img = image.img_to_array(img).astype(np.float32) / 255.0
    new_img = np.zeros_like(img)

    for i in range(0, new_img.shape[0], 32):
        for j in range(0, new_img.shape[1], 32):
            new_img[i:i + 32, j:j + 32] = model.predict(np.expand_dims(new_img[i:i + 32, j:j + 32], 0))

    new_img = image.array_to_img(new_img)
    new_img.save('Model1.png')