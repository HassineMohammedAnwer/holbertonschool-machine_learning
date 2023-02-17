#!/usr/bin/env python3
"""function returns matrices sum"""


def add_matrices2D(mat1, mat2):
    """matrix"""

    if len(mat1) != len(mat2) or any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
