#!/usr/bin/env python3
"""1. Inception Network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """builds an inception network
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should
    use a rectified linear activation (ReLU)
    Returns: the keras model
    """
    input_1 = K.Input(shape=(224, 224, 3))
    conv2d = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu'
    )(input_1)
    max_pooling2d = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv2d)
    conv2d_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(max_pooling2d)
    conv2d_2 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding='same',
        activation='relu'
    )(conv2d_1)
    max_pooling2d_1 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv2d_2)
    concatenate = inception_block(max_pooling2d_1, [64, 96, 128, 16, 32, 32])
    concatenate_1 = inception_block(concatenate, [128, 128, 192, 32, 96, 64])
    max_pooling2d_2 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(concatenate_1)
    concatenate_2 = inception_block(max_pooling2d_2, [192, 96, 208, 16, 48, 64])
    # lkn
    concatenate_3  = inception_block(concatenate_2, [160, 112, 224, 24, 64, 64])
    # avg prev from here
    concatenate_4 = inception_block(concatenate_3, [128, 128, 256, 24, 64, 64])
    concatenate_5 = inception_block(concatenate_4, [112, 144, 288, 32, 64, 64])
    # lkn
    concatenate_6 = inception_block(concatenate_5, [256, 160, 320, 32, 128, 128])
    # avg prev from here
    max_pooling2d_10 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(concatenate_6)
    concatenate_7 = inception_block(max_pooling2d_10, [256, 160, 320, 32, 128, 128])
    concatenate_8 = inception_block(concatenate_7, [384, 192, 384, 48, 128, 128])
    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding='valid'
    )(concatenate_8)
    dropout = K.layers.Dropout(rate=0.4)(average_pooling2d)
    dense =  K.layers.Dense(
        units=1000,
        activation='softmax',
        )(dropout)
    model = K.models.Model(inputs=input_1, outputs=dense)

    return model
