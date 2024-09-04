#!/usr/bin/env python3
"""concatenate"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """func concatenates two matrices along a specific axis"""
    return (np.concatenate((mat1, mat2), axis=axis))
