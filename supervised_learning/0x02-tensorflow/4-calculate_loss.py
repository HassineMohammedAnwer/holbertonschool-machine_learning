#!/usr/bin/env python3

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction

    Arguments:
    y -- placeholder for the labels of the input data
    y_pred -- tensor containing the network's predictions

    Returns:
    tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss
