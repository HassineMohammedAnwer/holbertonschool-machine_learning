#!/usr/bin/env python3
"""Specifity"""

import numpy as np


def specificity(confusion):
    """The specificity is defined as the ratio of true negatives to
    the total number of actual negatives"""
    tmp = np.sum(confusion, axis=1) + np.diag(confusion)
    true_negatives = np.sum(confusion) - np.sum(confusion, axis=0) - tmp
    false_positives = np.sum(confusion, axis=0) - np.diag(confusion)
    return true_negatives / (true_negatives + false_positives)
