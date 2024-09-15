#!/usr/bin/env python3
"""creates dropout layer """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """prev tensor = output of the previous layer
    n number of nodes the new layer
    activation function that should be used on the layer
    keep_prob is the proba that a node will be kept
    Returns: the output of the new layer"""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)(prev)

    if training:
        dropout = tf.nn.dropout(layer, rate=1 - keep_prob)

    return dropout
