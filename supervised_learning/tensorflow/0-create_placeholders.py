#!/usr/bin/env python3

"""function that return two holders"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """holders(x;y)"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
