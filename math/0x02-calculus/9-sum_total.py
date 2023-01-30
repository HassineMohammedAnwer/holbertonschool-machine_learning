#!/usr/bin/env python3

def summation_i_squared(n):
    """sum"""
    if n != int and n <= 0:
        return None
    res = 0
    for i in range(n + 1):
        res = res + i * i
    return (res)
