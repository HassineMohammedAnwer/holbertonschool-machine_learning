#!/usr/bin/env python3
"""Defines `create_layer`."""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ Creates a neural network layer  """
    layer = tf.keras.layers.Dense(
        activation=activation,
        name="layer",
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg')
    )
    return layer(prev)
