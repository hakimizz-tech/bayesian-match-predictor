import os
import pandas as pd


def save_to_csv(df, filename, data_dir):
    """Save a DataFrame to a CSV file in the data folder."""
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Loaded: {filename} -> {filepath}")
