#!/usr/bin/env python3
"""8. RMSProp"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """sets up the RMSProp optimization
    algorithm in TensorFlow"""
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            decay=beta2, epsilon=epsilon)
    return optimizer
