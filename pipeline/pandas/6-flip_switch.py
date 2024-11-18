#!/usr/bin/env python3
"""6. Flip it and Switch it"""


def flip_switch(df):
    """takes a pd.DataFrame and:
    Sorts the data in reverse chronological order.
    Transposes the sorted dataframe.
    Returns: the transformed pd.DataFrame
    """
    return df.sort_index(ascending=False).transpose()
