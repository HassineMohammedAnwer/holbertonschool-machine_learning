#!/usr/bin/env python3
"""function returns matrix's shape"""



def matrix_shape(matrix):
    """matrix"""
    res_matx = [len(matrix)]
    while isinstance(matrix[0], list):
        res_matx.append(len(matrix[0]))
        matrix = matrix[0]
    return res_matx
