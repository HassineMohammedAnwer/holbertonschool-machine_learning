#!/usr/bin/env python3
"""1. From Dictionary"""
import pandas as pd


"""creates a pd.DataFrame from a dictionary
The first column should be labeled First and have
__the values 0.0, 0.5, 1.0, and 1.5
The second column should be labeled Second and have
__the values one, two, three, four
The rows should be labeled A, B, C, and D, respectively
The pd.DataFrame should be saved into the variable df"""
df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"]
    },
    index=list("ABCD")
)
