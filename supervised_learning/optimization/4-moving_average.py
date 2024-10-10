#!/usr/bin/env python3
"""4. Moving Average"""

import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    v = 0
    v_t = []
    for i, x_t in enumerate(data, 1):
        v = v * beta + ((1 - beta) * x_t)
        v_t.append(v / (1 - beta ** i))
    return v_t
