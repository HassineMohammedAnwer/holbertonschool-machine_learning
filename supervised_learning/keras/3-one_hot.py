#!/usr/bin/env python3
"""converts a label vector to a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """last dimension of one-hot matrix must be num of classes
    Returns: one-hot matrix"""
    res = K.utils.to_categorical(labels, classes)
    return res
