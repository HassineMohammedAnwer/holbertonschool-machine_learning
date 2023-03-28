#!/usr/bin/env python3
"""qrgerqer"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Args:
        loss: the loss of the network's prediction.
        alpha: the learning rate.

    Returns:
        An operation that trains the network using gradient descent.
    """
    op = tf.train.GradientDescentOptimizer(alpha)
    train_op = op.minimize(loss)
    return train_op
