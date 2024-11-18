#!/usr/bin/env python3
"""4. To Numpy"""


def array(df):
    """  takes a pd.DataFrame as input and performs the following:
    df is a pd.DataFrame containing columns named High and Close.
    The function should select the last 10 rows of the High and Close columns.
    Convert these selected values into a numpy.ndarray.
    Returns: the numpy.ndarray"""
    return df[["High", "Close"]].tail(10).to_numpy()
