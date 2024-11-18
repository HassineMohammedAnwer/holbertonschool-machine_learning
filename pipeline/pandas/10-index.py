#!/usr/bin/env python3
"""10. Indexing"""


def index(df):
    """takes a pd.DataFrame and:
    Sets the Timestamp column as the index of the dataframe.
    Returns: the modified pd.DataFrame
    """
    return df.set_index('Timestamp')
