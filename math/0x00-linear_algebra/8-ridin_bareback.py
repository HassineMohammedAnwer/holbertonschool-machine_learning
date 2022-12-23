#!/usr/bin/env python3
"""function re"""


def mat_mul(mat1, mat2):
    """multiplicate matrices."""

    if (len(mat1[0]) != len(mat2)):
        return None
    li_a = []
    for i in range(len(mat1)):
        li_b = []
        for j in range(len(mat2[0])):
            x = 0
            for y in range(len(mat1[0])):
                x += mat1[i].copy()[y] * mat2[y].copy()[j]
            li_b.append(x)
        li_a.append(li_b)
    return li_a
