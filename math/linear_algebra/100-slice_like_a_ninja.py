#!/usr/bin/env python3
""" slices matrices """


def np_slice(matrix, axes={}):
    """
    slices a  matrix along specific axes.

    Args:
        matrix : The matrix to slice
        axes: dictionary where the key is an axis to slice along and
              the value is a tuple representing the slice to make along that axis

    Returns:
        A sliced matrix
    """
    slices = [slice(None)] * len(matrix.shape)
    for axis, slice_val in axes.items():
        slices[axis] = slice(*slice_val)
    return matrix[tuple(slices)]
