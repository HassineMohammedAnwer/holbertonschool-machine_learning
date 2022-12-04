#!/usr/bin/env python3
"""function concatenates 2 arrays"""


def cat_arrays(arr1, arr2):
    """array"""
    res_arr = []
    for i in range(len(arr1)):
        res_arr.append(arr1[i])
    for i in range(len(arr2)):
        res_arr.append(arr2[i])
    return res_arr
