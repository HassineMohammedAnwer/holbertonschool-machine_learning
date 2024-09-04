#!/usr/bin/env python3
""" func that imports numpy library """


def np_elementwise(mat1, mat2):
    """func performs element-wise addition,
    subtraction, multiplication, and division"""
    return ((mat1 + mat2), (mat1 - mat2),
            (mat1 * mat2), (mat1 / mat2))
