#!/usr/bin/env python3
"""pyth"""


def summation_i_squared(n):
    """sum"""
    if type(n) is not int:
        return None
    if n < 1:
        return None
    else:
        res = sum((map(lambda res: res ** 2, range(n + 1))))
        return res
