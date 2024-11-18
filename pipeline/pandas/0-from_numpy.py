#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray
    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    __and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame"""
    col = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    num_col = array.shape[1]
    df = pd.DataFrame(array, columns=col[:num_col])
    return df
