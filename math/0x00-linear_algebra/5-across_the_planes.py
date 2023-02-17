#!/usr/bin/env python3
"""function returns matrices sum"""


def add_matrices2D(mat1, mat2):
    """matrix"""
    if len(mat1[0]) != len(mat2[0]) :
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    res_matx = [[None for i in range(len(mat1))] for j in range(len(mat1[0]))]
    for i in range(len(mat1[0])):
        for j in range(len(mat1)):
            res_matx[i][j] = mat1[i][j] + mat2[i][j]
    return res_matx
