#!/usr/bin/env python3
"""afml aefo ^pkeaf """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 reg """

    initi = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizers = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            use_bias=True,
                            kernel_initializer=initi,
                            kernel_regularizer=regularizers)
    return layer(prev)
