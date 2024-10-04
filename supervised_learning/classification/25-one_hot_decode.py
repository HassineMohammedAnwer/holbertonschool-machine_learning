#!/usr/bin/env python3
""" One-Hot decode"""


import numpy as np


def one_hot_decode(one_hot):
    """  converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray):
        return None
    else:
        return np.argmax(one_hot, axis=0)
