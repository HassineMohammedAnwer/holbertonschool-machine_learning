#!/usr/bin/env python3
"""7. Sort"""


def high(df):
    """takes a pd.DataFrame and:
    Sorts it by the High price in descending order.
    Returns: the sorted pd.DataFrame"""
    return df['High'].sort_index(ascending=False)
