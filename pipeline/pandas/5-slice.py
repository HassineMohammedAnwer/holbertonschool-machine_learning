#!/usr/bin/env python3
"""5. Slice"""


def slice(df):
    """takes a pd.DataFrame and:
    Extracts the columns High, Low, Close, and Volume_BTC.
    Selects every 60th row from these columns.
    Returns: the sliced pd.DataFrame
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
