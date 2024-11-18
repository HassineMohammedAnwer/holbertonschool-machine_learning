#!/usr/bin/env python3
"""11. Concat"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """takes takes two pd.DataFrame objects and:
    Indexes both dataframes on their Timestamp columns.
    Includes all timestamps from df2 (bitstamp) up to and
    __including timestamp 1417411920.
    Concatenates the selected rows from df2 to the top of df1 (coinbase).
    Adds keys to the concatenated data, labeling the rows from df2 as
    __bitstamp and the rows from df1 as coinbase.
    Returns the concatenated pd.DataFrame.
    Returns: the modified pd.DataFrame
    """
    df2 = df2.loc[df2['Timestamp'] <= 1417411920]
    df1 = index(df1)
    df2 = index(df2)
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
