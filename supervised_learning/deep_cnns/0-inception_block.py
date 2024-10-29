#!/usr/bin/env python3
"""0. Inception Block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """builds an inception block 
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
    __F1 is the number of filters in the 1x1 convolution
    __F3R is the number of filters in the 1x1 convolution before the 3x3 convolution
    ___F3 is the number of filters in the 3x3 convolution
    __F5R is the number of filters in the 1x1 convolution before the 5x5 convolution
    __F5 is the number of filters in the 5x5 convolution
    __FPP is the number of filters in the 1x1 convolution after the max pooling
    All convolutions inside the inception block should use a rectified linear activation(ReLU)
    Returns: the concatenated output of the inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters

    l_F1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(A_prev)
    l_F3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(A_prev)
    l_F3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        activation='relu'
    )(l_F3R)
    l_F5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(A_prev)
    l_F5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        padding='same',
        activation='relu'
    )(l_F5R)
    l_pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=1,
        padding='same'
    )(A_prev)
    l_FPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(l_pool1)
    return K.layers.Concatenate()([l_F1, l_F3, l_F5, l_FPP])