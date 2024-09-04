#!/usr/bin/env python3
"""function re"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Cat matrices."""
    if (len(mat1[0]) == len(mat2[0])) and (axis == 0):
        res_matx = [i.copy() for i in mat1]
        res_matx += [i.copy() for i in mat2]
        return res_matx
    elif (len(mat1) == len(mat2)) and (axis == 1):
        res_matx = [mat1[j] + mat2[j] for j in range(len(mat1))]
        return res_matx
    return None
