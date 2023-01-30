#!/usr/bin/env python3

def summation_i_squared(n):
    """sum"""
    if type(n) is not int:
        return None
    if n < 1:
        return None
    if n == 1:
        return n
    else:
       return n * n + summation_i_squared(n-1)
