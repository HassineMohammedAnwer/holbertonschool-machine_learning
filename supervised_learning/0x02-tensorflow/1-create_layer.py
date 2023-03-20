#!/usr/bin/env python3
"""The task is to implement a function called create_layer that creates
  a new neural network layer. The function takes three arguments:
**prev: the tensor output of the previous layer.
**n: the number of nodes in the layer to create.
**activation: the activation function that the layer should use.
  The function should use the He et al. initialization for the layer weights,
  which is implemented using
  tf.keras.initializers.VarianceScaling(mode='fan_avg').
  Each layer should be given the name layer.

The function should return the tensor output of the layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """create layer"""
    pid = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    flake = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=deb, kernel_constraint=None,
                            name='layer')

    return flake(prev)
