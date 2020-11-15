import os
from tensorflow import keras
from tensorflow.keras import layers


class KNeuralNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = keras.Sequential()
        image_input_shape = (height, width, depth)
        if keras.backend.image_data_format() == "channels_first":
            image_input_shape = (depth, height, width)       
        # first layer
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", input_shape=image_input_shape))
        model.add(layers.Activation("relu"))
        # second layer
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # third layer
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # fourth layer
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # fifth layer
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # sixth layer
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        # seventh layer
        model.add(layers.Flatten())
        # eighth layer
        model.add(layers.Dense(classes))
        # last layer
        model.add(layers.Activation("softmax"))
        return model