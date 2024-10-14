#!/usr/bin/env python3
"""1. Correlation"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    std_d = np.sqrt(np.diag(C))
    denm = np.outer(std_d, std_d)
    correlation_m = C / denm
    return correlation_m
