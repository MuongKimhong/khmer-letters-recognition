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
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # fourth layer
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # fifth layer
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # sixth layer
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # seventh layer
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation('relu'))
        # layer 8
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # layer 9
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        # layer 10
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))
        # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        # layer 11
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation('relu'))
        # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        # layer 12
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation('relu'))
        # layer 13
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # layer 14
        model.add(layers.Flatten())
        # layer 15
        model.add(layers.Dense(classes))
        model.add(layers.Activation("softmax"))
        return model