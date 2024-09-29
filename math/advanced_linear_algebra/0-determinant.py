#!/usr/bin/env python3
""" task 0 """


def determinant(matrix):
    """ calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    n_cols = len(matrix[0])
    if n == 0:
        return 1
    if n == 1 and n_cols == 1:
        return matrix[0][0]
    if n != 1 and not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if n == 1 and n_cols == 0:
        return 1
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for col in range(n_cols):
        # Create submatrix (missing 1st and current column)
        submatrix = [row[0:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = matrix[0][col] * determinant(submatrix)
        # Add or subtract cofactor to the determinant
        det += cofactor if col % 2 == 0 else -cofactor

    return det
