#!/usr/bin/env python3
"""14. Batch Normalization Upgraded"""
import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """normalizes an unactivated output of a
    neural network using batch normalization"""
    k_i = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n,
                                  kernel_initializer=k_i,
                                  activation=None)(prev)
    gamma = tf.Variable(initial_value=tf.ones((n,)), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros((n,)), trainable=True)
    epsilon = 1e-7
    mean, variance = tf.nn.moments(layer, axes=[0])
    """mean = np.mean(layer, axis=0, keepdims=True)
    variance = np.var(layer, axis=0, keepdims=True)"""
    return activation(tf.nn.batch_normalization(
        layer, mean, variance, beta, gamma, epsilon, name=None
        ))
