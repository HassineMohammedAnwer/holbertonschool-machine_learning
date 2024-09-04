#!/usr/bin/env python3
"""function returns arrays sum """


def add_arrays(arr1, arr2):
    """array"""
    res_arr = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        res_arr.append(arr1[i] + arr2[i])
    return res_arr
