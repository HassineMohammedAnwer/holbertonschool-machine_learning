#!/usr/bin/env python3
"""preprocess data"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(path_file1, path_file2, save_path='preprocessed_data.csv'):
    """Convert Timestamp to datetime and set as index
    Merge datasets on Timestamp and sort
    Select relevant columns
    Handle missing values
    Normalize data
    Save preprocessed data
    """
    if not os.path.isfile(path_file1):
        raise FileNotFoundError(f"File {path_file1} doesn't exist.")
    if not os.path.isfile(path_file2):
        raise FileNotFoundError(f"File {path_file2} doesn't exist.")

    print(f"Loading data from {path_file1} and {path_file2}")
    df1 = pd.read_csv(path_file1)
    df2 = pd.read_csv(path_file2)

    df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], unit='s')
    df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], unit='s')
    df1.set_index('Timestamp', inplace=True)
    df2.set_index('Timestamp', inplace=True)

    combined_df = pd.concat([df1, df2], axis=0)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df.sort_index(inplace=True)

    relevant_columns = ['Open', 'High', 'Low', 'Close']
    combined_df = combined_df[relevant_columns]

    combined_df[relevant_columns] = combined_df[relevant_columns].ffill()
    combined_df.dropna(inplace=True)

    scaler = MinMaxScaler()
    combined_df[relevant_columns] = scaler.fit_transform(combined_df[relevant_columns])

    combined_df.to_csv(save_path)
    print(f"Preprocessed data saved to {save_path}")

    return combined_df


if __name__ == "__main__":
    # Preprocess the data
    preprocessed_data = preprocess_data("bitstamp.csv", "coinbase.csv", save_path='preprocessed_data.csv')
    print("Preprocessing completed!")
