#!/usr/bin/env python3
"""afml aefo ^pkeaf """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 reg """
    layer_weight = tf.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    L2_regularization = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=layer_weight,
                                  kernel_regularizer=L2_regularization)
    return layer(prev)
