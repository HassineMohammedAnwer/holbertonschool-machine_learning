#!/usr/bin/env python3
"""3. Rename"""
import pandas as pd


def rename(df):
    """ takes a pd.DataFrame as input and performs the following:
    df is a pd.DataFrame containing a column named Timestamp.
    The function should rename the Timestamp column to Datetime.
    Convert the timestamp values to datatime values
    Display only the Datetime and Close column
    Returns: the modified pd.DataFrame"""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    return df[['Datetime', 'Close']]
