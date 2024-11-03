#!/usr/bin/env python3
"""
4. ResNet-50"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds a projection block"""
    initializer = K.initializers.he_normal(seed=0)
    input_1 = K.Input(shape=(224, 224, 3))
    conv2d = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2,
        kernel_initializer=initializer
    )(input_1)
    batch_normalization = K.layers.BatchNormalization(axis=3)(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)
    max_pooling2d = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(activation)
    activation_3 = projection_block(max_pooling2d, [64, 64, 256], s=1)
    activation_6 = identity_block(activation_3, [64, 64, 256])
    activation_9 = identity_block(activation_6, [64, 64, 256])
    activation_12 = projection_block(activation_9, [128, 128, 512], s=2)
    activation_15 = identity_block(activation_12, [128, 128, 512])
    activation_18 = identity_block(activation_15, [128, 128, 512])
    activation_21 = identity_block(activation_18, [128, 128, 512])
    activation_24 = projection_block(activation_21, [256, 256, 1024], s=2)
    activation_27 = identity_block(activation_24, [256, 256, 1024])
    activation_30 = identity_block(activation_27, [256, 256, 1024])
    activation_33 = identity_block(activation_30, [256, 256, 1024])
    activation_36 = identity_block(activation_33, [256, 256, 1024])
    activation_39 = identity_block(activation_36, [256, 256, 1024])
    activation_42 = projection_block(activation_39, [512, 512, 2048], s=2)
    activation_45 = identity_block(activation_42, [512, 512, 2048])
    activation_48 = identity_block(activation_45, [512, 512, 2048])
    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        padding='same'
    )(activation_48)
    dense = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=initializer,
                           )(average_pooling2d)
    return K.models.Model(inputs=input_1, outputs=dense)
