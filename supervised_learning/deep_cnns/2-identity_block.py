#!/usr/bin/env python3
"""
2. Identity Block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """builds an identity block"""
    initializer = K.initializers.he_normal(seed=0)
    F11, F3, F12 = filters
    conv2d = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization = K.layers.BatchNormalization(axis=3)(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv2d_1 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        kernel_initializer=initializer
    )(activation)
    batch_normalization_1 = K.layers.BatchNormalization(axis=3)(conv2d_1)
    activation_1 = K.layers.Activation('relu')(batch_normalization_1)
    conv2d_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer
    )(activation_1)
    batch_normalization_2 = K.layers.BatchNormalization(axis=3)(conv2d_2)
    add = K.layers.Add()([batch_normalization_2, A_prev])
    activation_2 = K.layers.Activation('relu')(add)
    return activation_2
