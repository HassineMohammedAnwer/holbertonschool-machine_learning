#!/usr/bin/env python3
""" task 1. Minor """


def determinant(matrix):
    """ calculates the determinant of a matrix"""
    n = len(matrix)
    if (not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for col in range(len(matrix[0])):
        # Create submatrix (missing 1st and current column)
        submatrix = [row[0:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = matrix[0][col] * determinant(submatrix)
        # Add or subtract cofactor to the determinant
        det += cofactor if col % 2 == 0 else -cofactor

    return det


def minor(matrix):
    """ calculates the minor matrix of a matrix"""
    n = len(matrix)
    if (not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(len(matrix[i])):
            # create sub matrix removing i and j column
            sub_matrix_value = [row[:j] + row[j+1:] for row_idx, row
                                in enumerate(matrix) if row_idx != i]
            # ensure sub_matrix is a list of list
            sub_matrix_value = [row for row in sub_matrix_value if row]
            # check if sublist is empty
            if not sub_matrix_value:
                det_sub_matrix = 0
            else:
                det_sub_matrix = determinant(sub_matrix_value)
            minor_row.append(det_sub_matrix)
        minor_matrix.append(minor_row)

    return minor_matrix
