#!/usr/bin/env python3
"""
7. DenseNet-121"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ builds the DenseNet-121 architecture
    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and
    __a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    You may use:
    dense_block = __import__('5-dense_block').dense_block
    transition_layer = __import__('6-transition_layer').transition_layer
    Returns: the keras model"""
    initializer = K.initializers.he_normal(seed=0)
    input_1 = K.Input(shape=(224, 224, 3))
    layers = [6, 12, 24]
    batch_normalization = K.layers.BatchNormalization(axis=3)(input_1)
    activation = K.layers.Activation('relu')(batch_normalization)
    nb_filters = 2 * growth_rate
    conv2d = K.layers.Conv2D(filters=nb_filters,
                             kernel_size=7,
                             strides=2,
                             padding='same',
                             kernel_initializer=initializer,
                             )(activation)
    my_layer = K.layers.MaxPool2D(pool_size=3,
                                  padding='same',
                                  strides=2)(conv2d)
    for layer in layers:
        my_layer, nb_filters = dense_block(my_layer,
                                           nb_filters,
                                           growth_rate,
                                           layer)
        my_layer, nb_filters = transition_layer(my_layer,
                                                nb_filters,
                                                compression)
    my_layer, nb_filters = dense_block(my_layer, nb_filters, growth_rate, 16)
    my_layer = K.layers.AveragePooling2D(pool_size=7,
                                         padding='same')(my_layer)
    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)
    model = K.models.Model(inputs=input_1, outputs=my_layer)
    return model
