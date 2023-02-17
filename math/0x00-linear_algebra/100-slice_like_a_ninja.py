#!/usr/bin/env python3
""" func that imports numpy library """

import numpy as np


def np_slice(matrix, axes={}):
    """The function first initializes a list of slice(None) objects,
    which means that all elements along that axis will be included
    in the slice.
    Then, for each axis specified in the axes dictionary,
    it replaces the corresponding slice(None) object with the specified
    slice.

    Finally, it returns the sliced matrix using the numpy indexing syntax
    with a tuple of slices.

    The tuple(slices) syntax converts the list of slices into a tuple that
    can be used as an index for the matrix.
    """
    slices = [slice(None)] * matrix.ndim
    for axis, sl in axes.items():
        slices[axis] = slice(*sl)
    return matrix[tuple(slices)]
