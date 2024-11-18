#!/usr/bin/env python3
"""9. Fill"""


def fill(df):
    """takes a pd.DataFrame and:
    Removes the Weighted_Price column.
    Fills missing values in the Close column with the previous row’s value.
    Fills missing values in the High, Low, and Open columns with the
    __corresponding Close value in the same row.
    Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.
    Returns: the modified pd.DataFrame
    """
    df.drop(columns=['Weighted_Price'], inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)
    df[['High', 'Low', 'Open']] = (
        df[['High', 'Low', 'Open']].fillna(df['Close']))
    df[['Volume_(BTC)', 'Volume_(Currency)']] = (
        df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0))
    return df
