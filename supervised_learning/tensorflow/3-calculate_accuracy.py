#!/usr/bin/env python3
"""jvg;jg;jfjg"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction

    Arguments:
    y -- placeholder for the labels of the input data
    y_pred -- tensor containing the network's predictions

    Returns:
    tensor containing the decimal accuracy of the prediction
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
