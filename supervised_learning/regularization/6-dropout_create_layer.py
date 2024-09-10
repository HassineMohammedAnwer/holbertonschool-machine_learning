#!/usr/bin/env python3
"""creates dropout layer """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """prev tensor = output of the previous layer
    n number of nodes the new layer
    activation function that should be used on the layer
    keep_prob is the proba that a node will be kept
    Returns: the output of the new layer"""
    kernel_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=kernel_init,
                            kernel_regularizer=kernel_reg,
                            bias_regularizer=None,
                            name='layer')
    return layer(prev)
