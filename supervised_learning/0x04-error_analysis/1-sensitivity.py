#!/usr/bin/env python3
"""mmkjkj"""
import numpy as np


def sensitivity(confusion):
    """np.diag(confusion) returns the diagonal elements
    of the confusion matrix, which gives us the TP values for each class.
    np.sum(confusion, axis=1) - TP  calculates the FN values for each class.
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    return TP / (TP + FN)
