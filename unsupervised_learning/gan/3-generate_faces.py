#!/usr/bin/env python3
"""3. Generating faces"""
import tensorflow as tf
from tensorflow.keras import layers, models


def convolutional_GenDiscr():
    """sdqfs"""
    def generator():
        """sfvsfbv"""
        input_layer = layers.Input(shape=(16,))
        # Dense layer to expand the input to a larger dimension
        x = layers.Dense(2048, activation='tanh')(input_layer)
        # Reshape to a 2x2x512 tensor
        x = layers.Reshape((2, 2, 512))(x)
        # Upsample to 4x4
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)
        # Upsample to 8x8
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(16, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)
        # Upsample to 16x16
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(1, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        output_layer = layers.Activation('tanh')(x)

        return models.Model(input_layer, output_layer, name="generator")

    def get_discriminator():
        """eeth"""
        input_layer = layers.Input(shape=(16, 16, 1))
        # Convolutional layers with max pooling
        x = layers.Conv2D(32, kernel_size=3, padding='same')(input_layer)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)
        # Flatten and dense layer
        x = layers.Flatten()(x)
        output_layer = layers.Dense(1)(x)

        return models.Model(input_layer, output_layer, name="discriminator")

    return generator(), get_discriminator()
