#!/usr/bin/env python3
"""11. Concat"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """takes takes two pd.DataFrame objects and:
    Rearranges the MultiIndex so that Timestamp is the first level.
    Concatenates the bitstamp and coinbase tables from timestamps
    __1417411980 to 1417417980, inclusive.
    Adds keys to the data, labeling rows from df2 as bitstamp and
    __rows from df1 as coinbase.
    Ensures the data is displayed in chronological order.
    Returns: the concatenated pd.DataFrame
    """
    df1 = df1.loc[(df1['Timestamp'] >= 1417411980) &
                  (df1['Timestamp'] <= 1417417980)]
    df2 = df2.loc[(df2['Timestamp'] >= 1417411980) &
                  (df2['Timestamp'] <= 1417417980)]
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.reorder_levels([1, 0], axis=0)
    return df
