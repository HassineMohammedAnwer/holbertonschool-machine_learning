#!/usr/bin/env python3
"""
4-f1_score.py
"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calc the f1 score"""
    r_sensitivity = sensitivity(confusion)
    r_precision = precision(confusion)
    res = 2 * ((r_precision * r_sensitivity) / (r_precision + r_sensitivity))

    return res
