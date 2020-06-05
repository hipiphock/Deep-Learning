import tensorflow as tf
import matplotlib
from tensorflow import keras
import numpy as np

def model(input_shape):
    """
    input_shape: The height, width and channels as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = keras.layers.Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    X = keras.layers.Conv2D(32, (3, 3), activation='relu')(X)
    X = keras.layers.Conv2D(32, (3, 3), activation='relu')(X)
    X = keras.layers.BatchNormalization(axis = 3, name = 'bn0')(X)
    X = keras.layers.Activation('relu')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model

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
