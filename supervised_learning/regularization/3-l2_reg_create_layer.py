#!/usr/bin/env python3
"""afml aefo ^pkeaf """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 reg """
    l2_regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_regularizer
    )

    # Apply the layer to the previous tensor (prev)
    return layer(prev)
