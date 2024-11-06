#!/usr/bin/env python3
"""
5. Dense Block"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and
    a rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively"""
    initializer = K.initializers.he_normal(seed=0)
    i = 0
    while i < layers:
        my_layer = K.layers.BatchNormalization()(X)
        my_layer = K.layers.Activation('relu')(my_layer)
        my_layer = K.layers.Conv2D(filters=4*growth_rate,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)
        my_layer = K.layers.BatchNormalization()(my_layer)
        my_layer = K.layers.Activation('relu')(my_layer)
        my_layer = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)
        X = K.layers.concatenate([X, my_layer])
        nb_filters += growth_rate
        i += 1
    return X, nb_filters
