#!/usr/bin/env python3
"""13. Analyze"""
import pandas as pd


def analyze(df):
    """takes takes a pd.DataFrame and:
    Computes descriptive statistics for all columns except the Timestamp column.
    Returns a new pd.DataFrame containing these statistics.
    """
    df = df.drop(columns=['Timestamp'])
    return df.describe(include='all')
