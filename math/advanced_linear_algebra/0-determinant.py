#!/usr/bin/env python3
""" task 0 """


def determinant(matrix):
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")
    
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0:
        return 1

    if n != 1 and not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
