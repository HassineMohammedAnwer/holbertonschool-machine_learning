#!/usr/bin/env python3
"""function returns matrix's transpose"""


def matrix_transpose(matrix):
    """matrix"""
    """while isinstance(matrix[j], list):
        for i in matrix[j][c]:
            res_matx[j] = i
            j += 1
            c += 1"""
    res_matx = [[None
                 for i in range(len(matrix))] for j in range(len(matrix[0]))]
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            res_matx[i][j] = matrix[j][i]
    return res_matx
